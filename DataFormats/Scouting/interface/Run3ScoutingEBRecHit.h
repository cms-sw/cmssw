#ifndef DataFormats_Scouting_Run3ScoutingEBRecHit_h
#define DataFormats_Scouting_Run3ScoutingEBRecHit_h

#include <vector>

// Run-3 HLT-Scouting data format for EBRecHits
//
// IMPORTANT: any changes to Run3ScoutingEBRecHit must be backward-compatible !

class Run3ScoutingEBRecHit {
public:
  Run3ScoutingEBRecHit(float energy, float time, unsigned int detId, uint32_t flags)
      : energy_{energy}, time_{time}, detId_{detId}, flags_{flags} {}

  Run3ScoutingEBRecHit() : energy_{0}, time_{0}, detId_{0}, flags_{0} {}

  float energy() const { return energy_; }
  float time() const { return time_; }
  unsigned int detId() const { return detId_; }
  uint32_t flags() const { return flags_; }

private:
  float energy_;
  float time_;
  unsigned int detId_;
  uint32_t flags_;
};

using Run3ScoutingEBRecHitCollection = std::vector<Run3ScoutingEBRecHit>;

#endif
