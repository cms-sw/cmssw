#ifndef DataFormats_Scouting_Run3ScoutingHBHERecHit_h
#define DataFormats_Scouting_Run3ScoutingHBHERecHit_h

#include <vector>

// Updated Run 3 HLT-Scouting data format to include calo recHits information:
// - HBHERecHits collection (HCAL Barrel and Endcap)
// Saved information is specific to each hit type: energy and detId are available for HCAL recHits
//
// -- IMPORTANT: any changes to Run3ScoutingHBHERecHit must be backward-compatible!

class Run3ScoutingHBHERecHit {
public:
  Run3ScoutingHBHERecHit(float energy, unsigned int detId) : energy_{energy}, detId_{detId} {}

  Run3ScoutingHBHERecHit() : energy_{0}, detId_{0} {}

  float energy() const { return energy_; }
  unsigned int detId() const { return detId_; }

private:
  float energy_;
  unsigned int detId_;
};

using Run3ScoutingHBHERecHitCollection = std::vector<Run3ScoutingHBHERecHit>;

#endif
