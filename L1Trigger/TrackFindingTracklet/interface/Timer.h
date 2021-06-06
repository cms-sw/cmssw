#ifndef L1Trigger_TrackFindingTracklet_interface_Timer_h
#define L1Trigger_TrackFindingTracklet_interface_Timer_h

#include <cmath>
#include <chrono>

namespace trklet {

  class Timer {
  public:
    Timer() {}

    ~Timer() = default;

    void start();
    void stop();
    unsigned int ntimes() const { return ntimes_; }
    double avgtime() const { return ttot_ / ntimes_; }
    double rms() const { return sqrt((ttot_ * ttot_ - ttotsq_)) / ntimes_; }
    double tottime() const { return ttot_; }

  private:
    unsigned int ntimes_{0};
    double ttot_{0.0};
    double ttotsq_{0.0};

    std::chrono::high_resolution_clock::time_point tstart_;
  };
};  // namespace trklet
#endif
