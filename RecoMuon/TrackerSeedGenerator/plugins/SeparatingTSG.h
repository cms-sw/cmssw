#ifndef RecoMuon_TrackerSeedGenerator_SeparatingTSG_H
#define RecoMuon_TrackerSeedGenerator_SeparatingTSG_H

/** \class SeparatingTSG
 * Description:
 * composite TrackerSeedGenerator, which uses different TSG in different phase space of the track provided
 * concrete class must be implelemented (DualByEta ,...) to provide the TSG selection.
 *
 * \author Jean-Roch Vlimant
 */

#include "RecoMuon/TrackerSeedGenerator/plugins/CompositeTSG.h"

class TrackerTopology;

class SeparatingTSG : public CompositeTSG {
 public:
  SeparatingTSG(const edm::ParameterSet &pset);
  virtual ~SeparatingTSG();

  void trackerSeeds(const TrackCand&, const TrackingRegion&, const TrackerTopology *, BTSeedCollection &);

  virtual unsigned int selectTSG(const TrackCand&, const TrackingRegion&) =0;
 private:
  std::string theCategory;

};

#endif
