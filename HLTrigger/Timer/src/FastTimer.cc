// C++ headers
#include <chrono>
#include <iostream>     // cerr

#include "HLTrigger/Timer/interface/FastTimer.h"

FastTimer::FastTimer() :
  m_duration(Duration::zero()),
  m_start(),
  m_stop(),
  m_status(Status::kStopped)
{ }

// start the timer - only if it was not running
void FastTimer::start() {
  if (m_status == Status::kStopped) {
    m_start    = Clock::now();
  //m_stop     = Clock::time_point();   // FIXME re-enable this once there is a better way to fill the interpath data
    m_status   = Status::kRunning;
  } else {
    std::cerr << "attempting to start an already running timer" << std::endl;
  }
}

// stop the timer - only if it was running
void FastTimer::stop() {
  if (m_status == Status::kRunning) {
    m_stop     = Clock::now();
    m_duration += std::chrono::duration_cast<Duration>(m_stop - m_start);
    m_status   = Status::kStopped;
  }
}

// reset the timer
void FastTimer::reset() {
  m_duration = Duration::zero();
  m_start    = Clock::time_point();
  m_stop     = Clock::time_point();
  m_status   = Status::kStopped;
}

// read the accumulated time
FastTimer::Duration FastTimer::value() const {
  return m_duration;
}

// if the timer is stopped, read the accumulate time
// if the timer is running, also add the time up to "now" 
FastTimer::Duration FastTimer::untilNow() const {
  return m_duration + ( (m_status == Status::kRunning) ? std::chrono::duration_cast<Duration>(Clock::now() - m_start) : Duration::zero() );
}

FastTimer::Status FastTimer::status() const {
  return m_status;
}

FastTimer::Clock::time_point const & FastTimer::getStartTime() const {
  return m_start;
}

FastTimer::Clock::time_point const & FastTimer::getStopTime() const {
  return m_stop;
}

void FastTimer::setStartTime(FastTimer::Clock::time_point const & time) {
  m_start = time;
}

void FastTimer::setStopTime(FastTimer::Clock::time_point const & time) {
  m_stop = time;
}
