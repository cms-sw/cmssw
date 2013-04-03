#ifndef benchmark_h
#define benchmark_h

// C++ headers
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <chrono>


static constexpr unsigned int SIZE = 1000000;


double average(std::vector<double> const & values) {
  double sum = 0;
  for (size_t i = 0; i < values.size(); ++i)
    sum += values[i];
  return (sum / values.size());
}

double sigma(std::vector<double> const & values) {
  if (values.size() > 1) {
    double sum = 0;
    double avg = average(values); 
    for (size_t i = 0; i < values.size(); ++i)
      sum += (values[i] - avg) * (values[i] - avg);
    return std::sqrt( sum / (values.size()-1) );
  } else {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

double median(std::vector<double> const & values) {
  if (not values.empty()) {
    std::vector<double> v = values;
    std::sort( v.begin(), v.end() );
    return v[v.size() / 2];
  } else {
    return std::numeric_limits<double>::quiet_NaN();
  }
}


class BenchmarkBase {
public:
  BenchmarkBase() = default;

  explicit BenchmarkBase(std::string const & d) :
    description(d)
  { }

  virtual ~BenchmarkBase() { };

  // perform the measurements
  virtual void sample() = 0;

  void measure() {
    sample();
    start = std::chrono::high_resolution_clock::now();
    sample();
    stop  = std::chrono::high_resolution_clock::now();
  }

  // extract the characteristics of the timer from the measurements
  virtual void compute() = 0;

  // print a report
  virtual void report() = 0;

protected:
  std::chrono::high_resolution_clock::time_point    start;
  std::chrono::high_resolution_clock::time_point    stop;

  // measured per-call overhead
  double        overhead;
  
  // measured resolution, in seconds
  double        resolution_min;         // smallest of the steps
  double        resolution_median;      // median of the steps
  double        resolution_average;     // average of the steps
  double        resolution_avg_sig;     // sigma of the average
  double        resolution_sigma;       // sigma of the average

  // description
  std::string   description;
};


template <typename C>
class Benchmark : public BenchmarkBase {
public:
  typedef C                                 clock_type;
  typedef typename clock_type::time_point   time_point;

  Benchmark(std::string const & d) : 
    BenchmarkBase(d),
    values()
  {
  }

  // take SIZE measurements
  void sample() {
    for (unsigned int i = 0; i < SIZE; ++i)
      values[i] = clock_type::now();
  }

  // return the delta between two time_points, expressed in seconds
  double delta(const time_point & start, const time_point & stop) const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
  }

  // extract the characteristics of the timer from the measurements
  void compute() {
    // per-call overhead
    overhead = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() / SIZE;

    // resolution (min, median and average of the increments)
    std::vector<double> steps;
    steps.reserve(SIZE);
    for (unsigned int i = 0; i < SIZE-1; ++i) {
      double step = delta(values[i], values[i + 1]);
      if (step > 0)
        steps.push_back(step);
    }
    std::sort( steps.begin(), steps.end() );
    if (not steps.empty()) {
      // measure resolution as the median of the steps
      resolution_median  = steps[steps.size() / 2];

      // measure resolution as the first non-zero step
      resolution_min     = steps.front();

      // measure the sigma of the steps
      resolution_sigma   = sigma(steps);
      if (resolution_sigma < 1.e-10)
        resolution_sigma = 1.e-10;

      // measure average of the steps, and its sigma
      resolution_average = average(steps);
      resolution_avg_sig = resolution_sigma / sqrt(steps.size());

      // take into account limited accuracy
      if (resolution_avg_sig < 1.e-10)
        resolution_avg_sig = 1.e-10;
    }
  }

  // print a report
  void report() {
    std::cout << std::setprecision(1) << std::fixed;
    std::cout << "Performance of " << description << std::endl;
    std::cout << "\tAverage time per call: " << std::right << std::setw(10) << overhead    * 1e9 << " ns" << std::endl;
    if (not std::chrono::treat_as_floating_point<typename clock_type::rep>::value) {
      typename clock_type::duration             tick(1);
      std::chrono::duration<double, std::nano>  ns = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(tick);
      std::cout << "\tClock tick period:     " << std::right << std::setw(10) << ns.count() << " ns" << std::endl;
    } else {
      std::cout << "\tClock tick period:     " << std::right << std::setw(10) << "n/a" << std::endl;
    }
    std::cout << "\tMeasured resolution:   " << std::right << std::setw(10) << resolution_min  * 1e9 << " ns (median: " << resolution_median * 1e9 << " ns) (sigma: " << resolution_sigma * 1e9 << " ns) (average: " << resolution_average * 1e9 << " +/- " << resolution_avg_sig * 1e9 << " ns)" << std::endl;

    /*
    // warm up the cache
    measure_work(1000, false);

    std::cout << "\tTiming measurements (1k):" << std::endl;
    for (int i =0; i <10; ++i)
      measure_work(1000);
    std::cout << "\tTiming measurements (10k):" << std::endl;
    for (int i =0; i <10; ++i)
      measure_work(10000);
    std::cout << "\tTiming measurements (100k):" << std::endl;
    for (int i =0; i <10; ++i)
      measure_work(100000);
    */

    std::cout << std::endl;
  }

  // run some workload size times, taking a measurement around it
  void measure_work(int size, bool verbose = true) const {
    std::vector<double> times(size, 0.);
    for (int i = 0; i < size; ++i) {
      typename clock_type::time_point t0 = clock_type::now();
     
      volatile double x = M_PI;
      for (int j = 0; j < 100; ++j) {
        x = std::sqrt(x) * std::sqrt(x);
      }

      typename clock_type::time_point t1 = clock_type::now();
      times[i] = delta(t0, t1);
    }

    std::sort(times.begin(), times.end());

    double avg = average(times);
    double sig = sigma(times);
    double min = times.front();
    double max = times.back();
    double med = times[times.size() / 2]; 
    if (verbose) {
      std::cout << "\t\t[" << std::setw(10) << min * 1e9 << ".." << std::setw(10) << max * 1e9 << "] ns";
      std::cout << "\tavg: " << std::setw(10) << avg * 1e9 << " ns";
      std::cout << "\tmed: " << std::setw(10) << med * 1e9 << " ns";
      std::cout << "\tsig: " << std::setw(10) << sig * 1e9 << " ns";
    }

    // remove tails (values larger than twice the median)
    auto limit = std::upper_bound(times.begin(), times.end(), med * 2);
    std::vector<double> filtered(times.begin(), limit);

    avg = average(filtered);
    sig = sigma(filtered);
    min = filtered.front();
    max = filtered.back();
    med = filtered[filtered.size() / 2];
    if (verbose) {
      std::cout << "\t| [" << std::setw(10) << min * 1e9 << ".." << std::setw(10) << max * 1e9 << "] ns";
      std::cout << "\tavg: " << std::setw(10) << avg * 1e9 << " ns";
      std::cout << "\tmed: " << std::setw(10) << med * 1e9 << " ns";
      std::cout << "\tsig: " << std::setw(10) << sig * 1e9 << " ns";
    }

    std::cout << std::endl;
  }

protected:
  time_point    values[SIZE];

};

#endif //benchmark_h
