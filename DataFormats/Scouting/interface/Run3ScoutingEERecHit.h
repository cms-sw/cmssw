#ifndef DataFormats_Scouting_Run3ScoutingEERecHit_h
#define DataFormats_Scouting_Run3ScoutingEERecHit_h

#include <vector>

// Updated Run 3 HLT-Scouting data format to include calo recHits information:
// - EERecHits collection (ECAL Endcap)
// Saved information is specific to each hit type: energy, time, and detId are available for EE recHits
//
// IMPORTANT: any changes to Run3ScoutingEERecHit must be backward-compatible !

class Run3ScoutingEERecHit {
public:
  Run3ScoutingEERecHit(float energy, float time, unsigned int detId) : energy_{energy}, time_{time}, detId_{detId} {}

  Run3ScoutingEERecHit() : energy_{0}, time_{0}, detId_{0} {}

  float energy() const { return energy_; }
  float time() const { return time_; }
  unsigned int detId() const { return detId_; }

private:
  float energy_;
  float time_;
  unsigned int detId_;
};

using Run3ScoutingEERecHitCollection = std::vector<Run3ScoutingEERecHit>;

#endif
