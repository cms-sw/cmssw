#ifndef RecoMuon_TrackerSeedGenerator_CombinedTSG_H
#define RecoMuon_TrackerSeedGenerator_CombinedTSG_H

#include "RecoMuon/TrackerSeedGenerator/plugins/CompositeTSG.h"

class CombinedTSG : public CompositeTSG {
 public:
  CombinedTSG(const edm::ParameterSet &pset);
  ~CombinedTSG();

  void trackerSeeds(const TrackCand&, const TrackingRegion&, BTSeedCollection &);
  /// clean from shared hits 
  virtual std::vector<TrajectorySeed > cleanBySharedInput(const std::vector<TrajectorySeed>&,const std::vector<TrajectorySeed> &);

 private:
  std::string theCategory;
  bool firstTime;
};

#endif
