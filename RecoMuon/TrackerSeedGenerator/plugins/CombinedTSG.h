#ifndef RecoMuon_TrackerSeedGenerator_CombinedTSG_H
#define RecoMuon_TrackerSeedGenerator_CombinedTSG_H

/** \class CombinedTSG
 * Description:
 * CompositeTSG (TrackerSeedGenerator) which combines (with configurable duplicate removal) the output of different TSG.
 *
 * \author Jean-Roch Vlimant, Alessandro Grelli
*/

#include "RecoMuon/TrackerSeedGenerator/plugins/CompositeTSG.h"

class CombinedTSG : public CompositeTSG {
 public:
  CombinedTSG(const edm::ParameterSet &pset);
  ~CombinedTSG();

  /// provide the seeds from the TSGs: must be overloaded
  void trackerSeeds(const TrackCand&, const TrackingRegion&, BTSeedCollection &);
  /// clean from shared hits 
  virtual std::vector<TrajectorySeed > cleanBySharedInput(const std::vector<TrajectorySeed>&,const std::vector<TrajectorySeed> &);

 private:
  std::string theCategory;
  bool firstTime;
};

#endif
