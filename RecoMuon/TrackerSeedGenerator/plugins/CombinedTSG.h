#ifndef RecoMuon_TrackerSeedGenerator_CombinedTSG_H
#define RecoMuon_TrackerSeedGenerator_CombinedTSG_H

#include "RecoMuon/TrackerSeedGenerator/plugins/CompositeTSG.h"

class CombinedTSG : public CompositeTSG {
 public:
  CombinedTSG(const edm::ParameterSet &pset);
  ~CombinedTSG();

  void trackerSeeds(const TrackCand&, const TrackingRegion&, BTSeedCollection &);
 private:
  std::string theCategory;

};

#endif
