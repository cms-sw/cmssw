#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <boost/foreach.hpp>
// for timespec and clock_gettime
#include <time.h>
// for timeval and gettimeofday
#include <sys/time.h>
// for timeval and getrusage
#include <sys/resource.h>
// for omp_get_wtime
#include <omp.h>
// for times and tms, and sysconf
#include <sys/times.h>
#include <unistd.h>

#define SIZE 1000000


class TimerInterface {
public:
  // perform the measurements
  virtual void measure(void) = 0;

  // extract from characteristics of the timer from the measurements
  virtual void compute(void) = 0;

  // print a report
  virtual void report(void) = 0;
};


template <typename T>
class TimerBase : public TimerInterface {
public:
  typedef T time_type;

  // default ctor, initialize all members
  TimerBase() :
    values( new time_type[2*SIZE+1] ),
    granularity( 0. ),
    resolution( 0. ),
    resolution_mean( 0. ),
    resolution_sigma( 0. ),
    overhead( 0. ),
    ticks_per_second( 1. ),
    description()
  {
  }

  // return the delta between two time_type, expressed in seconds
  double delta(const time_type & start, const time_type & stop) {
    // generic implementation, see below for template specializations
    return (double) (stop - start) / ticks_per_second;
  }

  // extract from characteristics of the timer from the measurements
  void compute() {
    // per-call overhead
    overhead = delta(values[SIZE], values[2*SIZE]) / (double) SIZE;

    // resolution (median of the increments)
    std::vector<double> steps;
    steps.reserve(SIZE + 1);
    for (unsigned int i = 0; i < SIZE; ++i) {
      double step = delta(values[SIZE + i], values[SIZE + i + 1]);
      if (step > 0)
        steps.push_back(step);
    }
    std::sort( steps.begin(), steps.end() );
    if (not steps.empty()) {
      // measure resolution as the median of the steps
      resolution = steps[steps.size() / 2];

      // measure average and sigma of the steps
      double n    = steps.size();
      double sum  = 0.;
      for (size_t i = 0; i < steps.size(); ++i)
        sum  += steps[i];
      resolution_mean = sum / n;
      if (steps.size() > 1) {
        double sum2 = 0.;
        for (size_t i = 0; i < steps.size(); ++i)
          sum2 += (steps[i] - resolution_mean) * (steps[i] - resolution_mean);
        resolution_sigma = std::sqrt( sum2 / n );
      }
    }
  }

  // print a report
  void report() {
    std::cout << std::setprecision(0) << std::fixed;
    std::cout << "Performance of " << description << std::endl;
    std::cout << "\tAverage time per call: " << std::right << std::setw(10) << overhead    * 1e9 << " ns" << std::endl;
    std::cout << "\tReported resolution:   " << std::right << std::setw(10) << granularity * 1e9 << " ns" << std::endl;
    std::cout << "\tMeasured resolution:   " << std::right << std::setw(10) << resolution  * 1e9 << " ns (" << resolution_mean * 1e9 << " +/- " << resolution_sigma * 1e9 << " ns)" << std::endl;
    /*
    std::cout << "\tSteps:" << std::endl;
    for (unsigned int i = 0; i < SIZE; ++i) {
      double step = delta(values[SIZE + i], values[SIZE + i + 1]);
      if (step > 0)
        std::cout << "\t\t" << std::right << std::setw(10) << delta(values[SIZE + i], values[SIZE + i + 1]) * 1e9 << " ns" << std::endl;
    }
    */
    std::cout << std::endl;
  }

protected:
  time_type *   values;
  double        granularity;            // the reported resolution, in seconds
  double        resolution;      // the measured resolution, in seconds (median of the steps)
  double        resolution_mean;        // the measured resolution, in seconds (mean of the steps)
  double        resolution_sigma;       // the measured resolution, in seconds (sigma of the mean)
  double        overhead;               // the measured per-call overhead, in seconds
  double        ticks_per_second;       // optinally used by the delta() method
  std::string   description;
};


// timespec has a ns resolution
template <>
double TimerBase<timespec>::delta(const timespec & start, const timespec & stop) {
  if (stop.tv_nsec > start.tv_nsec)
    return (double) (stop.tv_sec - start.tv_sec) + (double) (stop.tv_nsec - start.tv_nsec) / (double) 1e9;
  else
    return (double) (stop.tv_sec - start.tv_sec) - (double) (start.tv_nsec - stop.tv_nsec) / (double) 1e9;
}

// timeval has a us resolution
template <>
double TimerBase<timeval>::delta(const timeval & start, const timeval & stop) {
  if (stop.tv_usec > start.tv_usec)
    return (double) (stop.tv_sec - start.tv_sec) + (double) (stop.tv_usec - start.tv_usec) / (double) 1e6;
  else
    return (double) (stop.tv_sec - start.tv_sec) - (double) (start.tv_usec - stop.tv_usec) / (double) 1e6;
}

// clock_t can have different resolutions !
template <>
double TimerBase<clock_t>::delta(const clock_t & start, const clock_t & stop) {
  return (double) (stop-start) / (double) ticks_per_second;
}

#ifdef __linux
// clock_gettime(CLOCK_THREAD_CPUTIME_ID)
class TimerClockGettimeThread : public TimerBase<timespec> {
public:
  TimerClockGettimeThread() {
    description = "clock_gettime(CLOCK_THREAD_CPUTIME_ID)";

    timespec value;
    clock_getres(CLOCK_THREAD_CPUTIME_ID, & value);
    granularity = value.tv_sec + value.tv_nsec / (double) 1e9;
  }

  void measure() {
    for (unsigned int i = 0; i <= 2*SIZE; ++i)
      clock_gettime(CLOCK_THREAD_CPUTIME_ID, values+i);
  }
};
#endif // __linux

#ifdef __linux
// clock_gettime(CLOCK_PROCESS_CPUTIME_ID)
class TimerClockGettimeProcess : public TimerBase<timespec> {
public:
  TimerClockGettimeProcess() {
    description = "clock_gettime(CLOCK_PROCESS_CPUTIME_ID)";

    timespec value;
    clock_getres(CLOCK_PROCESS_CPUTIME_ID, & value);
    granularity = value.tv_sec + value.tv_nsec / (double) 1e9;
  }

  void measure() {
    for (unsigned int i = 0; i <= 2*SIZE; ++i)
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, values+i);
  }
};
#endif // __linux

#ifdef __linux
// clock_gettime(CLOCK_REALTIME)
class TimerClockGettimeRealtime : public TimerBase<timespec> {
public:
  TimerClockGettimeRealtime() {
    description = "clock_gettime(CLOCK_REALTIME)";

    timespec value = { 0, 0 };
    clock_getres(CLOCK_REALTIME, & value);
    granularity = value.tv_sec + value.tv_nsec / (double) 1e9;
  }

  void measure() {
    for (unsigned int i = 0; i <= 2*SIZE; ++i)
      clock_gettime(CLOCK_REALTIME, values+i);
  }
};
#endif // __linux

#ifdef __linux
// clock_gettime(CLOCK_MONOTONIC)
class TimerClockGettimeMonotonic : public TimerBase<timespec> {
public:
  TimerClockGettimeMonotonic() {
    description = "clock_gettime(CLOCK_MONOTONIC)";

    timespec value = { 0, 0 };
    clock_getres(CLOCK_MONOTONIC, & value);
    granularity = value.tv_sec + value.tv_nsec / (double) 1e9;
  }

  void measure() {
    for (unsigned int i = 0; i <= 2*SIZE; ++i)
      clock_gettime(CLOCK_MONOTONIC, values+i);
  }
};
#endif // __linux

// gettimeofday()
class TimerGettimeofday : public TimerBase<timeval> {
public:
  TimerGettimeofday() {
    description = "gettimeofday()";
    granularity = 1e-6;
  }

  void measure() {
    for (unsigned int i = 0; i <= 2*SIZE; ++i)
      gettimeofday(values+i, NULL);
  }
};


// getrusage(RUSAGE_SELF)
class TimerGetrusageSelf: public TimerBase<timeval> {
public:
  TimerGetrusageSelf() {
    description = "getrusage(RUSAGE_SELF)";
    granularity = 1e-6;
  }

  void measure() {
    rusage usage;
    for (unsigned int i = 0; i <= 2*SIZE; ++i) {
      getrusage(RUSAGE_SELF, &usage);
      values[i].tv_sec  = usage.ru_stime.tv_sec  + usage.ru_utime.tv_sec;
      values[i].tv_usec = usage.ru_stime.tv_usec + usage.ru_utime.tv_usec;
    }
  }
};


// getrusage(RUSAGE_SELF)
class TimerOMPGetWtime : public TimerBase<double> {
public:
  TimerOMPGetWtime() {
    description = "omp_get_wtime()";
    granularity = 1e-9;
  }

  void measure() {
    for (unsigned int i = 0; i <= 2*SIZE; ++i)
      values[i] = omp_get_wtime();
  }
};


// clock()
class TimerClock : public TimerBase<clock_t> {
public:
  TimerClock() {
    description = "clock()";
    ticks_per_second = CLOCKS_PER_SEC;
    granularity = 1. / ticks_per_second;
  }

  void measure() {
    for (unsigned int i = 0; i <= 2*SIZE; ++i)
      values[i] = clock();
  }
};


// times()
class TimerTimes : public TimerBase<clock_t> {
public:
  TimerTimes() {
    description = "times()";
    ticks_per_second = sysconf(_SC_CLK_TCK);
    granularity = 1. / ticks_per_second;
  }

  void measure() {
    tms value;
    for (unsigned int i = 0; i <= 2*SIZE; ++i) {
      times(& value);
      values[i] = value.tms_stime + value.tms_utime;
    }
  }
};



int main(void) {
  std::vector<TimerInterface *> timers;
#ifdef __linux
  timers.push_back(new TimerClockGettimeThread());
  timers.push_back(new TimerClockGettimeProcess());
  timers.push_back(new TimerClockGettimeRealtime());
  timers.push_back(new TimerClockGettimeMonotonic());
#endif // __linux
  timers.push_back(new TimerGettimeofday());
  timers.push_back(new TimerGetrusageSelf());
  timers.push_back(new TimerOMPGetWtime());
  timers.push_back(new TimerClock());
  timers.push_back(new TimerTimes());

  BOOST_FOREACH(TimerInterface * timer, timers) {
    timer->measure();
    timer->compute();
    timer->report();
  }

  return 0;
}
