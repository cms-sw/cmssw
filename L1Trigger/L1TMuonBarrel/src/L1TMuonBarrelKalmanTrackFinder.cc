#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanTrackFinder.h"



L1TMuonBarrelKalmanTrackFinder::L1TMuonBarrelKalmanTrackFinder(const edm::ParameterSet& iConfig): 
  verbose_(iConfig.getParameter<int>("verbose"))
{
  std::vector<int> sectors = iConfig.getParameter<std::vector<int> >("sectorsToProcess");
    for (const auto sector : sectors)
      sectors_.push_back(L1TMuonBarrelKalmanSectorProcessor(iConfig.getParameter<edm::ParameterSet>("sectorSettings"),sector));
}



L1TMuonBarrelKalmanTrackFinder::~L1TMuonBarrelKalmanTrackFinder() {}

L1MuKBMTrackCollection L1TMuonBarrelKalmanTrackFinder::process(L1TMuonBarrelKalmanAlgo* trackMaker, const L1MuKBMTCombinedStubRefVector& stubsAll,int bx) {


  L1MuKBMTrackCollection pretracks;
  for (auto& sector: sectors_) {
    L1MuKBMTrackCollection tmp = sector.process(trackMaker,stubsAll,bx);
    if (!tmp.empty())
      pretracks.insert(pretracks.end(),tmp.begin(),tmp.end());
  } 
  if (verbose_) {
    printf(" -----Track Finder Kalman Tracks (Uncleaned!)-----\n");
    for (const auto& track1 :pretracks)
    printf("Kalman Track charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f\n",track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.approxChi2(),track1.pt(),track1.ptUnconstrained()); 
  }
  return pretracks;

}
