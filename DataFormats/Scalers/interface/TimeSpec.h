
#ifndef DataFormats_Scalers_TimeSpec_h
#define DataFormats_Scalers_TimeSpec_h

#include <ctime>

class TimeSpec {

public:

  TimeSpec() :
    tv_sec_(0),
    tv_nsec_(0) {}

  TimeSpec(long tv_sec, long tv_nsec) :
    tv_sec_(tv_sec),
    tv_nsec_(tv_nsec) {}

  TimeSpec(timespec const& ts) :
    tv_sec_(static_cast<long>(ts.tv_sec)),
    tv_nsec_(static_cast<long>(ts.tv_nsec)) {}

  long tv_sec() const { return tv_sec_; } 
  long tv_nsec() const { return tv_nsec_; }

  void set_tv_sec(long value) { tv_sec_ = value; } 
  void set_tv_nsec(long value) { tv_nsec_ = value; }

  timespec get_timespec() const;

private:

  long tv_sec_;   // seconds
  long tv_nsec_;  // nanoseconds
};

#endif
