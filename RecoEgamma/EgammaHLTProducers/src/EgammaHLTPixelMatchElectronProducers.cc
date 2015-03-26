// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTPixelMatchElectronProducers
// 
/**\class EgammaHLTPixelMatchElectronProducers RecoEgamma/ElectronProducers/src/EgammaHLTPixelMatchElectronProducers.cc

 Description: EDProducer of HLT Electron objects

*/
//
// Original Author: Monica Vazquez Acosta (CERN)
// $Id: EgammaHLTPixelMatchElectronProducers.cc,v 1.3 2009/01/28 17:07:00 ghezzi Exp $
//
//

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTPixelMatchElectronProducers.h"

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTPixelMatchElectronAlgo.h"

#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

#include <iostream>

using namespace reco;
 
EgammaHLTPixelMatchElectronProducers::EgammaHLTPixelMatchElectronProducers(const edm::ParameterSet& iConfig) : conf_(iConfig) {

  consumes<TrackCollection>(conf_.getParameter<edm::InputTag>("TrackProducer"));
  consumes<GsfTrackCollection>(conf_.getParameter<edm::InputTag>("GsfTrackProducer"));
  consumes<BeamSpot>(conf_.getParameter<edm::InputTag>("BSProducer"));

  //create algo
  algo_ = new EgammaHLTPixelMatchElectronAlgo(conf_, consumesCollector());

  //register your products
  produces<ElectronCollection>();
}


EgammaHLTPixelMatchElectronProducers::~EgammaHLTPixelMatchElectronProducers() {
  delete algo_;
}

void EgammaHLTPixelMatchElectronProducers::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("TrackProducer"), edm::InputTag("hltEleAnyWP80CleanMergedTracks"));
  desc.add<edm::InputTag>(("GsfTrackProducer"), edm::InputTag(""));
  desc.add<bool>(("UseGsfTracks"), false);
  desc.add<edm::InputTag>(("BSProducer"), edm::InputTag("hltOnlineBeamSpot")); 
  descriptions.add(("hltEgammaHLTPixelMatchElectronProducers"), desc);  
}

// ------------ method called to produce the data  ------------
void EgammaHLTPixelMatchElectronProducers::produce(edm::StreamID sid, edm::Event& e, const edm::EventSetup& iSetup) const {
  // Update the algorithm conditions
  algo_->setupES(iSetup);  
  
  // Create the output collections   
  std::auto_ptr<ElectronCollection> pOutEle(new ElectronCollection);
  
  // invoke algorithm
    algo_->run(e,*pOutEle);

  // put result into the Event
    e.put(pOutEle);
}


