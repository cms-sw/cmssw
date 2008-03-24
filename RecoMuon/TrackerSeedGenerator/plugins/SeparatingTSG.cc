#include "RecoMuon/TrackerSeedGenerator/plugins/SeparatingTSG.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SeparatingTSG::SeparatingTSG(const edm::ParameterSet &pset):CompositeTSG(pset){}

SeparatingTSG::~SeparatingTSG(){}

void SeparatingTSG::trackerSeeds(const TrackCand & muonTrackCand, const TrackingRegion& region, std::vector<TrajectorySeed> & result){
  uint sel = selectTSG(muonTrackCand,region);
  LogDebug(theCategory)<<"choosing: "<<theNames[sel]<<", at index ["<<sel<<"]";
  if(theTSGs[sel]) {
    std::vector<TrajectorySeed>  tmpResult;
    theTSGs[sel]->trackerSeeds(muonTrackCand,region,tmpResult);
    result.insert(result.end(),tmpResult.begin(),tmpResult.end());
    if(theHistos[sel]) theHistos[sel]->Fill(tmpResult.size());
  }
}
