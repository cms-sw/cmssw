#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanSectorProcessor.h"



L1TMuonBarrelKalmanSectorProcessor::L1TMuonBarrelKalmanSectorProcessor(const edm::ParameterSet& iConfig,int sector): 
  verbose_(iConfig.getParameter<int>("verbose")),
  sector_(sector)
{
  std::vector<int> wheels = iConfig.getParameter<std::vector<int> >("wheelsToProcess");
    for (const auto wheel : wheels)
      regions_.push_back(L1TMuonBarrelKalmanRegionModule(iConfig.getParameter<edm::ParameterSet>("regionSettings"),wheel,sector));
}



L1TMuonBarrelKalmanSectorProcessor::~L1TMuonBarrelKalmanSectorProcessor() {}

L1MuKBMTrackCollection L1TMuonBarrelKalmanSectorProcessor::process(L1TMuonBarrelKalmanAlgo* trackMaker, const L1MuKBMTCombinedStubRefVector& stubsAll,int bx) {


  L1MuKBMTrackCollection pretracks;
  for (auto& region: regions_) {
    L1MuKBMTrackCollection tmp = region.process(trackMaker,stubsAll,bx);
    if (tmp.size()>0)
      pretracks.insert(pretracks.end(),tmp.begin(),tmp.end());
  } 

  L1MuKBMTrackCollection out =trackMaker->cleanAndSort(pretracks,3);




  return out;

}
