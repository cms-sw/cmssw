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

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include <iostream>
#include <string>
#include <memory>

#include "EgammaHLTPixelMatchElectronAlgo.h"

class EgammaHLTPixelMatchElectronProducers : public edm::stream::EDProducer<> {
public:
  explicit EgammaHLTPixelMatchElectronProducers(const edm::ParameterSet& conf);

  void produce(edm::Event& e, const edm::EventSetup& c) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  EgammaHLTPixelMatchElectronAlgo algo_;
  const edm::EDPutTokenT<reco::ElectronCollection> token_;
};

using namespace reco;

EgammaHLTPixelMatchElectronProducers::EgammaHLTPixelMatchElectronProducers(const edm::ParameterSet& iConfig)
    : algo_(iConfig, consumesCollector()), token_(produces<ElectronCollection>()) {
  consumes<TrackCollection>(iConfig.getParameter<edm::InputTag>("TrackProducer"));
  consumes<GsfTrackCollection>(iConfig.getParameter<edm::InputTag>("GsfTrackProducer"));
  consumes<BeamSpot>(iConfig.getParameter<edm::InputTag>("BSProducer"));
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
void EgammaHLTPixelMatchElectronProducers::produce(edm::Event& e, const edm::EventSetup& iSetup) {
  // Update the algorithm conditions
  algo_.setupES(iSetup);

  // Create the output collections
  ElectronCollection outEle;

  // invoke algorithm
  algo_.run(e, outEle);

  // put result into the Event
  e.emplace(token_, std::move(outEle));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTPixelMatchElectronProducers);
