// C++ headers
#include <chrono>
#include <iostream>
#include <vector>
#include <string>

#include "HLTrigger/Timer/interface/FastTimer.h"

FastTimer::FastTimer() :
  m_start(),
  m_stop(),
  m_duration(Duration::zero()),
  m_state(State::kStopped)
{ }

// start the timer - only if it was not running
void FastTimer::start() {
  if (m_state == State::kStopped) {
    m_start    = Clock::now();
    m_stop     = Clock::time_point();
    m_duration = Duration::zero();
    m_state    = State::kRunning;
  } else {
    std::cerr << "attempting to start a " << describe() << " timer" << std::endl;
  }
}

// stop the timer - only if it was running
void FastTimer::stop() {
  if (m_state == State::kRunning) {
    m_stop     = Clock::now();
    m_duration += std::chrono::duration_cast<Duration>(m_stop - m_start);
    m_state    = State::kStopped;
  } else {
    std::cerr << "attempting to stop a " << describe() << " timer" << std::endl;
  }
}

// pause the timer - only if it was running
void FastTimer::pause() {
  if (m_state == State::kRunning) {
    m_stop     = Clock::now();
    m_duration += std::chrono::duration_cast<Duration>(m_stop - m_start);
    m_state    = State::kPaused;
  } else {
    std::cerr << "attempting to pause a " << describe() << " timer" << std::endl;
  }
}

// resume the timer - only if it was not running
void FastTimer::resume() {
  if (m_state == State::kPaused) {
    m_start    = Clock::now();
    m_stop     = Clock::time_point();
    m_state    = State::kRunning;
  } else {
    std::cerr << "attempting to resume a " << describe() << " timer" << std::endl;
  }
}

// reset the timer
void FastTimer::reset() {
  m_start    = Clock::time_point();
  m_stop     = Clock::time_point();
  m_duration = Duration::zero();
  m_state    = State::kStopped;
}

// read the accumulated time
FastTimer::Duration FastTimer::value() const {
  return m_duration;
}

// read the accumulated time, in seconds
double FastTimer::seconds() const {
  return std::chrono::duration_cast<std::chrono::duration<double>>(m_duration).count();
}

// if the timer is stopped, read the accumulate time
// if the timer is running, also add the time up to "now" 
FastTimer::Duration FastTimer::untilNow() const {
  return m_duration + ( (m_state == State::kRunning) ? std::chrono::duration_cast<Duration>(Clock::now() - m_start) : Duration::zero() );
}

double FastTimer::secondsUntilNow() const {
  return std::chrono::duration_cast<std::chrono::duration<double>>(untilNow()).count();
}

// return the current state
FastTimer::State FastTimer::state() const {
  return m_state;
}

// descrbe the current state
std::string const & FastTimer::describe() const {
  static const std::vector<std::string> states{ "stopped", "running", "paused", "unknown" };

  switch (m_state) {
    case FastTimer::State::kStopped:
    case FastTimer::State::kRunning:
    case FastTimer::State::kPaused:
      return states[static_cast<unsigned int>(m_state)];

    default:
      return states.back();
  }
}

FastTimer::Clock::time_point const & FastTimer::getStartTime() const {
  return m_start;
}

FastTimer::Clock::time_point const & FastTimer::getStopTime() const {
  return m_stop;
}

void FastTimer::setStartTime(FastTimer::Clock::time_point const & time) {
  if (m_state == State::kStopped) {
    m_start    = time;
    m_stop     = Clock::time_point();
    m_duration = Duration::zero();
    m_state    = State::kRunning;
  } else {
    std::cerr << "attempting to start a " << describe() << " timer" << std::endl;
  }
}

void FastTimer::setStopTime(FastTimer::Clock::time_point const & time) {
  if (m_state == State::kRunning) {
    m_stop     = time;
    m_duration += std::chrono::duration_cast<Duration>(m_stop - m_start);
    m_state    = State::kStopped;
  } else {
    std::cerr << "attempting to stop a " << describe() << " timer" << std::endl;
  }
}
