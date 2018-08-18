#include "L1Trigger/L1TTrackMatch/interface/L1TTrackerPlusBarrelStubsSectorProcessor.h"



L1TTrackerPlusBarrelStubsSectorProcessor::L1TTrackerPlusBarrelStubsSectorProcessor(const edm::ParameterSet& iConfig, int sector): 
  verbose_(iConfig.getParameter<int>("verbose")),
  propagation_(iConfig.getParameter<std::vector<double> > ("propagationConstants"))
{
  sector_ = sector;
  //find the previous and the next processor
  if (sector==11) {
    previousSector_ = 10;
    nextSector_=0;
  } 
  else if (sector==0) {
    previousSector_ = 11;
    nextSector_=1;
  } 
  else {
    previousSector_ = sector-1;
    nextSector_=sector+1;

  }

}



L1TTrackerPlusBarrelStubsSectorProcessor::~L1TTrackerPlusBarrelStubsSectorProcessor() {}

std::vector<l1t::L1TkMuonParticle> L1TTrackerPlusBarrelStubsSectorProcessor::process(const TrackPtrVector& tracks,const L1MuKBMTCombinedStubRefVector& stubsAll) {

  //Output collction
  std::vector<l1t::L1TkMuonParticle> out;
  //First thing first. Keep only the stubs on this processor
  L1MuKBMTCombinedStubRefVector stubs;

  for (const auto& stub: stubsAll) 
    if (stub->scNum()==sector_ || stub->scNum()==previousSector_ ||stub->scNum()==nextSector_)
      stubs.push_back(stub);
  
  //Next loop on tracks
  for (const auto& track : tracks) {
    //You need some logic to see if track is in this sector here:

    
    //Create a muon particle from the track:
    l1t::L1TkMuonParticle::LorentzVector vec(track->getMomentum().x(),
					track->getMomentum().y(),
					track->getMomentum().z(),
					track->getMomentum().mag());
    l1t::L1TkMuonParticle muon (vec,track);


    //pt phi and eta can be accesed by vec.phi(),vec.eta(),vec.pt()
    //propagate and match here 


    //You only need to add stubs to the muons  at this stage
    //To do that just do:
    //muon.addBarrelStub(stub);
     

    //for now just add it
    out.push_back(muon);
  }

  return out;

}



