// -*- C++ -*-
//
// Package:    EgammaElectronProducers
// Class:      GsfElectronProducer
// 
/**\class GsfElectronProducer RecoEgamma/ElectronProducers/src/GsfElectronProducer.cc

 Description: EDProducer of GsfElectron objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: GsfElectronProducer.cc,v 1.5 2008/03/04 17:04:00 uberthon Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "GsfElectronProducer.h"

#include <iostream>

using namespace reco;
 
GsfElectronProducer::GsfElectronProducer(const edm::ParameterSet& iConfig) 
{
  //register your products
  produces<GsfElectronCollection>();

  //create algo
  algo_ = new
    GsfElectronAlgo(iConfig,
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

GsfElectronProducer::~GsfElectronProducer()
{
  delete algo_;
}

void GsfElectronProducer::beginJob(edm::EventSetup const&iSetup) 
{     
}

// ------------ method called to produce the data  ------------
void GsfElectronProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{
  algo_->setupES(iSetup);  

  // Create the output collections   
  std::auto_ptr<GsfElectronCollection> pOutEle(new GsfElectronCollection);
  
  // invoke algorithm
  algo_->run(e,*pOutEle);

  // put result into the Event
  e.put(pOutEle);
  
}


