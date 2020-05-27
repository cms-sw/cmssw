#include "L1Trigger/TrackFindingTracklet/interface/Timer.h"

using namespace trklet;

void Timer::start() { tstart_ = std::chrono::high_resolution_clock::now(); }
void Timer::stop() {
  auto tstop = std::chrono::high_resolution_clock::now();
  double tmp = std::chrono::duration<double>(tstop - tstart_).count();
  ttot_ += tmp;
  ttotsq_ += tmp * tmp;
  ntimes_++;
}
