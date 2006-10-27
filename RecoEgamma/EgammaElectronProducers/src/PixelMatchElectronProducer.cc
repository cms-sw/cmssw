// -*- C++ -*-
//
// Package:    EgammaElectronProducers
// Class:      PixelMatchElectronProducer
// 
/**\class PixelMatchElectronProducer RecoEgamma/ElectronProducers/src/PixelMatchElectronProducer.cc

 Description: EDProducer of Electron objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: PixelMatchElectronProducer.cc,v 1.8 2006/10/27 15:04:05 uberthon Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaElectronProducers/interface/PixelMatchElectronProducer.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchElectronAlgo.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"

#include <iostream>

using namespace reco;
 
PixelMatchElectronProducer::PixelMatchElectronProducer(const edm::ParameterSet& iConfig) : conf_(iConfig)
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


PixelMatchElectronProducer::~PixelMatchElectronProducer()
{
  delete algo_;
}

void PixelMatchElectronProducer::beginJob(edm::EventSetup const&iSetup) 
{     
  algo_->setupES(iSetup,conf_);  
}

// ------------ method called to produce the data  ------------
void PixelMatchElectronProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{

  // Create the output collections   
  std::auto_ptr<PixelMatchGsfElectronCollection> pOutEle(new PixelMatchGsfElectronCollection);
  
  // invoke algorithm
    algo_->run(e,*pOutEle);

  // put result into the Event
    e.put(pOutEle);
  
}


