#include "RecoMuon/TrackerSeedGenerator/plugins/CombinedTSG.h"

CombinedTSG::CombinedTSG(const edm::ParameterSet & par) : CompositeTSG(par) {
  theCategory = "combinedTSG";
}

CombinedTSG::~CombinedTSG(){}

void CombinedTSG::trackerSeeds(const TrackCand & muonTrackCand, const TrackingRegion& region, std::vector<TrajectorySeed> & result){
  //run all the seed generators registered
  for (uint iTSG=0; iTSG!=theTSGs.size();iTSG++){
    theTSGs[iTSG]->trackerSeeds(muonTrackCand,region,result);
  }
  //add some seed cleaning may be
}
