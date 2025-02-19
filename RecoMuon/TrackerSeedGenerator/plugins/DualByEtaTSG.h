#ifndef RecoMuon_TrackerSeedGenerator_DualByEtaTSG_H
#define RecoMuon_TrackerSeedGenerator_DualByEtaTSG_H

/** \class DualByEtaTSG
 * Description:
 * SeparatingTSG (TrackerSeedGenerator) which make a simple, dual selection based on the momentum pseudo rapidity of the input track.
 *
 * \author Jean-Roch vlimant, Adam Everett
 */

#include "RecoMuon/TrackerSeedGenerator/plugins/SeparatingTSG.h"


class DualByEtaTSG : public SeparatingTSG{
 public:
  DualByEtaTSG(const edm::ParameterSet &pset);

  /// decide the TSG depending on the absolute value of momentum eta of the track. Return value is 0 or 1.
  unsigned int selectTSG(const TrackCand&, const TrackingRegion&);
 private:
  std::string theCategory;
  double theEtaSeparation;
};

#endif
