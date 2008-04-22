#ifndef RecoMuon_TrackerSeedGenerator_DualByEtaTSG_H
#define RecoMuon_TrackerSeedGenerator_DualByEtaTSG_H

#include "RecoMuon/TrackerSeedGenerator/plugins/SeparatingTSG.h"


class DualByEtaTSG : public SeparatingTSG{
 public:
  DualByEtaTSG(const edm::ParameterSet &pset);

  uint selectTSG(const TrackCand&, const TrackingRegion&);
 private:
  std::string theCategory;
  double theEtaSeparation;
};

#endif
