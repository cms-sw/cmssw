// -*- C++ -*-
//
// Package:    EgammaElectronProducers
// Class:      GlobalGsfElectronProducer
// 
/**\class GlobalGsfElectronProducer RecoEgamma/ElectronProducers/src/GlobalGsfElectronProducer.cc

 Description: EDProducer of GsfElectron objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: GlobalGsfElectronProducer.cc,v 1.6 2008/04/15 21:31:13 charlot Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GlobalGsfElectronAlgo.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "GlobalGsfElectronProducer.h"

#include <iostream>

using namespace reco;
 
GlobalGsfElectronProducer::GlobalGsfElectronProducer(const edm::ParameterSet& iConfig) 
{
  //register your products
  produces<GsfElectronCollection>();

  //create algo
  algo_ = new
    GlobalGsfElectronAlgo(iConfig,
		    iConfig.getParameter<double>("maxEOverPBarrel"),
		    iConfig.getParameter<double>("maxEOverPEndcaps"),
		    iConfig.getParameter<double>("minEOverPBarrel"),
		    iConfig.getParameter<double>("minEOverPEndcaps"),
		    iConfig.getParameter<double>("maxDeltaEta"),
		    iConfig.getParameter<double>("maxDeltaPhi"),
		    iConfig.getParameter<bool>("highPtPreselection"),
		    iConfig.getParameter<double>("highPtMin"),
		    iConfig.getParameter<bool>("applyEtaCorrection"),
		    iConfig.getParameter<bool>("applyAmbResolution")
		    );

}

GlobalGsfElectronProducer::~GlobalGsfElectronProducer()
{
  delete algo_;
}

void GlobalGsfElectronProducer::beginJob(edm::EventSetup const&iSetup) 
{     
}

// ------------ method called to produce the data  ------------
void GlobalGsfElectronProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{
  algo_->setupES(iSetup);  

  // Create the output collections   
  std::auto_ptr<GsfElectronCollection> pOutEle(new GsfElectronCollection);
  
  // invoke algorithm
  algo_->run(e,*pOutEle);

  // put result into the Event
  e.put(pOutEle);
  
}


