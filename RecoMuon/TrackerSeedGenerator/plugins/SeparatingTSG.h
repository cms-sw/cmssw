#ifndef RecoMuon_TrackerSeedGenerator_SeparatingTSG_H
#define RecoMuon_TrackerSeedGenerator_SeparatingTSG_H

#include "RecoMuon/TrackerSeedGenerator/plugins/CompositeTSG.h"

class SeparatingTSG : public CompositeTSG {
 public:
  SeparatingTSG(const edm::ParameterSet &pset);
  virtual ~SeparatingTSG();

  void trackerSeeds(const TrackCand&, const TrackingRegion&, BTSeedCollection &);

  virtual uint selectTSG(const TrackCand&, const TrackingRegion&) =0;
 private:
  std::string theCategory;

};

#endif
