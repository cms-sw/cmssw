// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonRefProducer
//
//
// Original Author:  Dmytro Kovalskyi
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "RecoMuon/MuonIdentification/plugins/MuonRefProducer.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

MuonRefProducer::MuonRefProducer(const edm::ParameterSet& iConfig) {
  theReferenceCollection_ = iConfig.getParameter<edm::InputTag>("ReferenceCollection");
  muonToken_ = consumes<reco::MuonCollection>(theReferenceCollection_);

  type_ = muon::TMLastStation;  // default type
  std::string type = iConfig.getParameter<std::string>("algorithmType");
  if (type != "TMLastStation")
    edm::LogWarning("MuonIdentification")
        << "Unknown algorithm type is requested: " << type << "\nUsing the default one.";

  minNumberOfMatches_ = iConfig.getParameter<int>("minNumberOfMatchedStations");
  maxAbsDx_ = iConfig.getParameter<double>("maxAbsDx");
  maxAbsPullX_ = iConfig.getParameter<double>("maxAbsPullX");
  maxAbsDy_ = iConfig.getParameter<double>("maxAbsDy");
  maxAbsPullY_ = iConfig.getParameter<double>("maxAbsPullY");
  maxChamberDist_ = iConfig.getParameter<double>("maxChamberDistance");
  maxChamberDistPull_ = iConfig.getParameter<double>("maxChamberDistancePull");

  std::string arbitrationType = iConfig.getParameter<std::string>("arbitrationType");
  if (arbitrationType == "NoArbitration")
    arbitrationType_ = reco::Muon::NoArbitration;
  else if (arbitrationType == "SegmentArbitration")
    arbitrationType_ = reco::Muon::SegmentArbitration;
  else if (arbitrationType == "SegmentAndTrackArbitration")
    arbitrationType_ = reco::Muon::SegmentAndTrackArbitration;
  else {
    edm::LogWarning("MuonIdentification")
        << "Unknown arbitration type is requested: " << arbitrationType << "\nUsing the default one";
    arbitrationType_ = reco::Muon::SegmentAndTrackArbitration;
  }
  produces<edm::RefVector<std::vector<reco::Muon>>>();
}

MuonRefProducer::~MuonRefProducer() {}

void MuonRefProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto outputCollection = std::make_unique<edm::RefVector<std::vector<reco::Muon>>>();

  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(muonToken_, muons);

  // loop over input collection
  for (unsigned int i = 0; i < muons->size(); ++i)
    if (muon::isGoodMuon((*muons)[i],
                         type_,
                         minNumberOfMatches_,
                         maxAbsDx_,
                         maxAbsPullX_,
                         maxAbsDy_,
                         maxAbsPullY_,
                         maxChamberDist_,
                         maxChamberDistPull_,
                         arbitrationType_))
      outputCollection->push_back(edm::RefVector<std::vector<reco::Muon>>::value_type(muons, i));
  iEvent.put(std::move(outputCollection));
}
