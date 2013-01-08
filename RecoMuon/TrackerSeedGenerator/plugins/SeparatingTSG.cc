#include "RecoMuon/TrackerSeedGenerator/plugins/SeparatingTSG.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

SeparatingTSG::SeparatingTSG(const edm::ParameterSet &pset):CompositeTSG(pset){}

SeparatingTSG::~SeparatingTSG(){}

void SeparatingTSG::trackerSeeds(const TrackCand & muonTrackCand, const TrackingRegion& region, const TrackerTopology *tTopo,
				 std::vector<TrajectorySeed> & result){
  unsigned int sel = selectTSG(muonTrackCand,region);
  LogDebug(theCategory)<<"choosing: "<<theNames[sel]<<", at index ["<<sel<<"]";
  if(theTSGs[sel]) {
    std::vector<TrajectorySeed>  tmpResult;
    theTSGs[sel]->trackerSeeds(muonTrackCand,region,tTopo,tmpResult);
    result.insert(result.end(),tmpResult.begin(),tmpResult.end());
  }
}
