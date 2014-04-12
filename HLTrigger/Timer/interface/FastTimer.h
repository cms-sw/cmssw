#ifndef FastTimer_h
#define FastTimer_h

// C++ headers
#include <chrono>

class FastTimer {
public:
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::nanoseconds           Duration;
  enum class State {
    kStopped,
    kRunning,
    kPaused
  };

  FastTimer();

  void start();
  void pause();
  void resume();
  void stop();
  void reset();

  Duration value() const;
  Duration untilNow() const;
  double seconds() const;
  double secondsUntilNow() const;
  State state() const;
  Clock::time_point const & getStartTime() const;
  Clock::time_point const & getStopTime() const;
  void setStartTime(Clock::time_point const &);
  void setStopTime(Clock::time_point const &);

private:
  std::string const & describe() const;

  Clock::time_point m_start;
  Clock::time_point m_stop;
  Duration          m_duration;
  State             m_state;
};

#endif // ! FastTimer_h
