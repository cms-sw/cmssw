#include "RecoMuon/TrackerSeedGenerator/plugins/CombinedTSG.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>
#include <vector>

//constructor
CombinedTSG::CombinedTSG(const edm::ParameterSet & par) : CompositeTSG(par) {
  theCategory = "CombinedTSG";
}

//destructor
CombinedTSG::~CombinedTSG(){
 //
}

void CombinedTSG::trackerSeeds(const TrackCand & muonTrackCand, const TrackingRegion& region, std::vector<TrajectorySeed> & result){
  //run all the seed generators registered

  for (uint iTSG=0; iTSG!=theTSGs.size();iTSG++){
    if(theTSGs[iTSG]) {
      std::vector<TrajectorySeed>  tmpResult;
      theTSGs[iTSG]->trackerSeeds(muonTrackCand,region,tmpResult);
      //vector of seeds
      result.insert(result.end(),tmpResult.begin(),tmpResult.end());
      if(theHistos[iTSG]) theHistos[iTSG]->Fill(tmpResult.size());
    }
  }
}
