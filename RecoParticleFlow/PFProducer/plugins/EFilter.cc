// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
// #include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

// #include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoParticleFlow/PFProducer/plugins/EFilter.h"


// #include "FastSimulation/Event/interface/FSimEvent.h"
// #include "FastSimulation/Event/interface/FSimTrack.h"
// #include "FastSimulation/Event/interface/FSimVertex.h"

// #include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
// #include "FastSimulation/Particle/interface/ParticleTable.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
// #include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"


using namespace edm;
using namespace std;

EFilter::EFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed

  inputTagParticles_ = consumes<std::vector<reco::PFSimParticle> >(iConfig.getParameter<InputTag>("particles"));

  minE_ = iConfig.getUntrackedParameter<double>("minE",-1);
  maxE_ = iConfig.getUntrackedParameter<double>("maxE",999999);
  minEt_ = iConfig.getUntrackedParameter<double>("minEt",-1);
  maxEt_ = iConfig.getUntrackedParameter<double>("maxEt",999999);

  LogWarning("PFProducer")<<"EFilter : will filter particles with "
			  <<minE_<<" < E < "<<maxE_<<" and "<<minEt_<<" < Et <"<<maxEt_<<endl;

}


EFilter::~EFilter() {
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  // delete mySimEvent;


}


bool
EFilter::filter(edm::Event& iEvent, 
		const edm::EventSetup& iSetup) {



  Handle<std::vector<reco::PFSimParticle> > particles;
  iEvent.getByToken(inputTagParticles_, particles);

  if( !particles->empty() ) {
    // take first trajectory point of first particle (the mother)
    const reco::PFTrajectoryPoint& tp = (*particles)[0].trajectoryPoint(0);
    
    const math::XYZTLorentzVector& mom = tp.momentum();

    double e = mom.E();
    double et = mom.Et(); 

    if( e >= minE_  && e<= maxE_ && 
	et>= minEt_ && et<= maxEt_ ) {
      cout<<"ok "<<e<<endl;
      return true;
    }
    else {
      cout<<"bad "<<e<<endl;	
      return false;
    }
  }


  
  return true;
}

