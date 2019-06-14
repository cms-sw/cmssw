//*****************************************************************************
// File:      EgammaElectronTkNumIsolationProducer.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaElectronTkNumIsolationProducer.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"

EgammaElectronTkNumIsolationProducer::EgammaElectronTkNumIsolationProducer(const edm::ParameterSet& config)
    :  // use configuration file to setup input/output collection names
      electronProducer_(config.getParameter<edm::InputTag>("electronProducer")),

      trackProducer_(config.getParameter<edm::InputTag>("trackProducer")),
      beamspotProducer_(config.getParameter<edm::InputTag>("BeamspotProducer")),

      ptMin_(config.getParameter<double>("ptMin")),
      intRadiusBarrel_(config.getParameter<double>("intRadiusBarrel")),
      intRadiusEndcap_(config.getParameter<double>("intRadiusEndcap")),
      stripBarrel_(config.getParameter<double>("stripBarrel")),
      stripEndcap_(config.getParameter<double>("stripEndcap")),
      extRadius_(config.getParameter<double>("extRadius")),
      maxVtxDist_(config.getParameter<double>("maxVtxDist")),
      drb_(config.getParameter<double>("maxVtxDistXY")) {
  //register your products
  produces<edm::ValueMap<int>>();
}

EgammaElectronTkNumIsolationProducer::~EgammaElectronTkNumIsolationProducer() {}

void EgammaElectronTkNumIsolationProducer::produce(edm::StreamID sid,
                                                   edm::Event& iEvent,
                                                   const edm::EventSetup& iSetup) const {
  // Get the  filtered objects
  edm::Handle<reco::GsfElectronCollection> electronHandle;
  iEvent.getByLabel(electronProducer_, electronHandle);

  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(trackProducer_, tracks);
  const reco::TrackCollection* trackCollection = tracks.product();

  //prepare product
  auto isoMap = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler filler(*isoMap);
  std::vector<int> retV(electronHandle->size(), 0);

  //get beamspot
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByLabel(beamspotProducer_, beamSpotH);
  reco::TrackBase::Point beamspot = beamSpotH->position();

  ElectronTkIsolation myTkIsolation(extRadius_,
                                    intRadiusBarrel_,
                                    intRadiusEndcap_,
                                    stripBarrel_,
                                    stripEndcap_,
                                    ptMin_,
                                    maxVtxDist_,
                                    drb_,
                                    trackCollection,
                                    beamspot);

  for (unsigned int i = 0; i < electronHandle->size(); ++i) {
    int isoValue = myTkIsolation.getNumberTracks(&(electronHandle->at(i)));
    retV[i] = isoValue;
  }

  //fill and insert valuemap
  filler.insert(electronHandle, retV.begin(), retV.end());
  filler.fill();
  iEvent.put(std::move(isoMap));
}
