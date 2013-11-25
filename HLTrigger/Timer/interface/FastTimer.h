#ifndef FastTimer_h
#define FastTimer_h

// C++ headers
#include <chrono>

class FastTimer {
public:
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::nanoseconds           Duration;
  enum class Status {
    kStopped,
    kRunning
  };

  FastTimer();

  void start();
  void stop();
  void reset();

  Duration value() const;
  Duration untilNow() const;
  Status status() const;
  Clock::time_point const & getStartTime() const;
  Clock::time_point const & getStopTime() const;
  void setStartTime(Clock::time_point const &);
  void setStopTime(Clock::time_point const &);

private:
  Duration          m_duration;
  Clock::time_point m_start;
  Clock::time_point m_stop;
  Status            m_status;
};

#endif // ! FastTimer_h
