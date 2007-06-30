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
// $Id: PixelMatchElectronProducer.cc,v 1.2 2007/05/09 14:20:25 uberthon Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchElectronAlgo.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchElectronFwd.h"

#include "PixelMatchElectronProducer.h"

#include <iostream>

using namespace reco;
 
PixelMatchElectronProducer::PixelMatchElectronProducer(const edm::ParameterSet& iConfig) : conf_(iConfig)
{
  //register your products
  produces<PixelMatchElectronCollection>();

  //create algo
  algo_ = new
  PixelMatchElectronAlgo(iConfig.getParameter<double>("maxEOverPBarrel"),
			 iConfig.getParameter<double>("maxEOverPEndcaps"),
                         iConfig.getParameter<double>("minEOverPBarrel"),
			 iConfig.getParameter<double>("minEOverPEndcaps"),
			 iConfig.getParameter<double>("hOverEConeSize"),
			 iConfig.getParameter<double>("maxHOverE"),
			 iConfig.getParameter<double>("maxDeltaEta"),
			 iConfig.getParameter<double>("maxDeltaPhi"),
			 iConfig.getParameter<double>("ptCut"),
			 iConfig.getParameter<double>("highPtPreselection"),
			 iConfig.getParameter<double>("highPtMin"));

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
  std::auto_ptr<PixelMatchElectronCollection> pOutEle(new PixelMatchElectronCollection);
  
  // invoke algorithm
  // FIXME :template version to be implemented 
  //    algo_->run(e,*pOutEle);
  throw cms::Exception("Configuration")
          << "Creation of non-GsfElectrons is not implemented for the moment!!!"
          << std::endl << "Please use Gsf-version of cfg file to produce GsfElectrons!! ";

  // put result into the Event
  e.put(pOutEle);
  
}


