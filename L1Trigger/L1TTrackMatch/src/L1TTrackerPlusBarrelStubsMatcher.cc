#include "L1Trigger/L1TTrackMatch/interface/L1TTrackerPlusBarrelStubsMatcher.h"



L1TTrackerPlusBarrelStubsMatcher::L1TTrackerPlusBarrelStubsMatcher(const edm::ParameterSet& iConfig): 
  verbose_(iConfig.getParameter<int>("verbose"))
{
  //Create sector processors
  std::vector<int> sectors = iConfig.getParameter<std::vector<int> >("sectorsToProcess");
    for (const auto sector : sectors)
      sectors_.push_back(L1TTrackerPlusBarrelStubsSectorProcessor(iConfig.getParameter<edm::ParameterSet>("sectorSettings"),sector));
}



L1TTrackerPlusBarrelStubsMatcher::~L1TTrackerPlusBarrelStubsMatcher() {}

std::vector<l1t::L1TkMuonParticle> L1TTrackerPlusBarrelStubsMatcher::process(const TrackPtrVector& tracks,const L1MuKBMTCombinedStubRefVector& stubs) {


  std::vector<l1t::L1TkMuonParticle> preMuons;
  for (auto& sector: sectors_) {
    std::vector<l1t::L1TkMuonParticle> tmp = sector.process(tracks,stubs);
    if (!tmp.empty())
      preMuons.insert(preMuons.end(),tmp.begin(),tmp.end());
  } 

  //Clean muons from different processors
  std::vector<l1t::L1TkMuonParticle> muons = overlapClean(preMuons);

  return muons;

}



std::vector<l1t::L1TkMuonParticle> L1TTrackerPlusBarrelStubsMatcher::overlapClean(const std::vector<l1t::L1TkMuonParticle>& preMuons) {


  //Change this with the code cleaning logic
  return preMuons;

}
