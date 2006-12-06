// -*- C++ -*-
//
// Package:    EgammaElectronProducers
// Class:      PixelMatchGsfElectronProducer
// 
/**\class PixelMatchGsfElectronProducer RecoEgamma/ElectronProducers/src/PixelMatchGsfElectronProducer.cc

 Description: EDProducer of PixelMatchGsfElectron objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: PixelMatchGsfElectronProducer.cc,v 1.9 2006/10/27 16:34:21 uberthon Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoEgamma/EgammaElectronProducers/interface/PixelMatchGsfElectronProducer.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchElectronAlgo.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"
#include "DataFormats/TrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"

#include <iostream>

using namespace reco;
 
PixelMatchGsfElectronProducer::PixelMatchGsfElectronProducer(const edm::ParameterSet& iConfig) : conf_(iConfig)
{
  //register your products
  produces<PixelMatchGsfElectronCollection>();

  //create algo
  algo_ = new
  PixelMatchElectronAlgo(iConfig.getParameter<double>("maxEOverPBarrel"),
                     iConfig.getParameter<double>("maxEOverPEndcaps"),
                     iConfig.getParameter<double>("hOverEConeSize"),
                     iConfig.getParameter<double>("maxHOverE"),
                     iConfig.getParameter<double>("maxDeltaEta"),
                     iConfig.getParameter<double>("maxDeltaPhi"));

}


PixelMatchGsfElectronProducer::~PixelMatchGsfElectronProducer()
{
  delete algo_;
}

void PixelMatchGsfElectronProducer::beginJob(edm::EventSetup const&iSetup) 
{     
  algo_->setupES(iSetup,conf_);  
}

// ------------ method called to produce the data  ------------
void PixelMatchGsfElectronProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{

  // Create the output collections   
  std::auto_ptr<PixelMatchGsfElectronCollection> pOutEle(new PixelMatchGsfElectronCollection);
  
  // invoke algorithm
    algo_->run(e,*pOutEle);

  // put result into the Event
    e.put(pOutEle);
  
}


