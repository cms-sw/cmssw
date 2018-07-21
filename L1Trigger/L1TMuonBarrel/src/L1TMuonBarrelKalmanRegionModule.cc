#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanRegionModule.h"



L1TMuonBarrelKalmanRegionModule::L1TMuonBarrelKalmanRegionModule(const edm::ParameterSet& iConfig,int wheel,int sector): 
  verbose_(iConfig.getParameter<int>("verbose")),
  sector_(sector),
  wheel_(wheel)
{

  if (sector==11) {
    nextSector_=0;
    previousSector_ = 10;
  }
  else if (sector==0) {
    nextSector_=1;
    previousSector_ = 11;
  }
  else {
    nextSector_= sector+1;
    previousSector_ = sector-1;
  }

  switch(wheel) {

  case -2:
    nextWheel_=-1;
    break;

  case -1:
    nextWheel_=0;
    break;

  case 0:
    nextWheel_=999;
    break;

  case 1:
    nextWheel_=0;
    break;

  case 2:
    nextWheel_=1;
    break;

  default:
    nextWheel_=999;
    break;
  }
}



L1TMuonBarrelKalmanRegionModule::~L1TMuonBarrelKalmanRegionModule() {}

L1MuKBMTrackCollection L1TMuonBarrelKalmanRegionModule::process(L1TMuonBarrelKalmanAlgo* trackMaker, const L1MuKBMTCombinedStubRefVector& stubsAll,int bx) {
  L1MuKBMTCombinedStubRefVector stubs;
  L1MuKBMTCombinedStubRefVector seeds;
  L1MuKBMTrackCollection pretracks;
  for (const auto& stub : stubsAll) {
    if (stub->bxNum()!=bx)
      continue;

    if ((stub->scNum()==nextSector_ && stub->phi()>=-112)||(stub->scNum()==previousSector_ && stub->phi()<=111))
      continue;
    
    if (stub->whNum()==wheel_  && stub->scNum()==sector_) {
      seeds.push_back(stub);
      stubs.push_back(stub);
    }
    else if (stub->whNum()==wheel_  && (stub->scNum()==nextSector_||stub->scNum()==previousSector_ )) {
      stubs.push_back(stub);
    }
    else if (stub->whNum()==nextWheel_  && (stub->scNum()==nextSector_||stub->scNum()==previousSector_||stub->scNum()==sector_) ) {
      stubs.push_back(stub);
    }
  }

  for (const auto seed : seeds) {
    std::pair<bool,L1MuKBMTrack> trackInfo = trackMaker->chain(seed,stubs);
    if (trackInfo.first)
      pretracks.push_back(trackInfo.second);
  } 

  //  trackMaker->resolveEtaUnit(pretracks);
  L1MuKBMTrackCollection out =trackMaker->cleanAndSort(pretracks,2);
  if (verbose_) {
    printf(" -----Sector Processor Kalman Tracks-----\n");
    for (const auto& track1 :out)
      printf("Kalman Track charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f\n",track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.approxChi2(),track1.pt(),track1.ptUnconstrained()); 
  }


  return out;
}


