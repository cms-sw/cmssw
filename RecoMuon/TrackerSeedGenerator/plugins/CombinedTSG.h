#ifndef RecoMuon_TrackerSeedGenerator_CombinedTSG_H
#define RecoMuon_TrackerSeedGenerator_CombinedTSG_H

/** \class CombinedTSG
 * Description:
 * CompositeTSG (TrackerSeedGenerator) which combines (with configurable duplicate removal) the output of different TSG.
 *
 * \author Jean-Roch Vlimant, Alessandro Grelli
*/

#include "RecoMuon/TrackerSeedGenerator/plugins/CompositeTSG.h"
class TrackerTopology;

class CombinedTSG : public CompositeTSG {
 public:
  CombinedTSG(const edm::ParameterSet &pset);
  ~CombinedTSG();

  /// provide the seeds from the TSGs: must be overloaded
  void trackerSeeds(const TrackCand&, const TrackingRegion&, const TrackerTopology *, BTSeedCollection &);

 private:
  std::string theCategory;
};

#endif
