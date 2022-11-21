//
// Package:    MuonTimingProducer
// Class:      MuonTimingProducer
//
/**\class MuonTimingProducer MuonTimingProducer.cc RecoMuon/MuonIdentification/src/MuonTimingProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Piotr Traczyk, CERN
//         Created:  Mon Mar 16 12:27:22 CET 2009
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

#include "RecoMuon/MuonIdentification/plugins/MuonTimingProducer.h"
#include "RecoMuon/MuonIdentification/interface/TimeMeasurementSequence.h"

//
// constructors and destructor
//
MuonTimingProducer::MuonTimingProducer(const edm::ParameterSet& iConfig) {
  produces<reco::MuonTimeExtraMap>("combined");
  produces<reco::MuonTimeExtraMap>("dt");
  produces<reco::MuonTimeExtraMap>("csc");

  m_muonCollection = iConfig.getParameter<edm::InputTag>("MuonCollection");
  muonToken_ = consumes<reco::MuonCollection>(m_muonCollection);
  // Load parameters for the TimingFiller
  edm::ParameterSet fillerParameters = iConfig.getParameter<edm::ParameterSet>("TimingFillerParameters");
  theTimingFiller_ = new MuonTimingFiller(fillerParameters, consumesCollector());
}

MuonTimingProducer::~MuonTimingProducer() {
  if (theTimingFiller_)
    delete theTimingFiller_;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void MuonTimingProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto muonTimeMap = std::make_unique<reco::MuonTimeExtraMap>();
  reco::MuonTimeExtraMap::Filler filler(*muonTimeMap);
  auto muonTimeMapDT = std::make_unique<reco::MuonTimeExtraMap>();
  reco::MuonTimeExtraMap::Filler fillerDT(*muonTimeMapDT);
  auto muonTimeMapCSC = std::make_unique<reco::MuonTimeExtraMap>();
  reco::MuonTimeExtraMap::Filler fillerCSC(*muonTimeMapCSC);

  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(muonToken_, muons);

  unsigned int nMuons = muons->size();

  std::vector<reco::MuonTimeExtra> dtTimeColl(nMuons);
  std::vector<reco::MuonTimeExtra> cscTimeColl(nMuons);
  std::vector<reco::MuonTimeExtra> combinedTimeColl(nMuons);

  for (unsigned int i = 0; i < nMuons; ++i) {
    reco::MuonTimeExtra dtTime;
    reco::MuonTimeExtra cscTime;
    reco::MuonTime rpcTime;
    reco::MuonTimeExtra combinedTime;

    reco::MuonRef muonr(muons, i);

    theTimingFiller_->fillTiming(*muonr, dtTime, cscTime, rpcTime, combinedTime, iEvent, iSetup);

    dtTimeColl[i] = dtTime;
    cscTimeColl[i] = cscTime;
    combinedTimeColl[i] = combinedTime;
  }

  filler.insert(muons, combinedTimeColl.begin(), combinedTimeColl.end());
  filler.fill();
  fillerDT.insert(muons, dtTimeColl.begin(), dtTimeColl.end());
  fillerDT.fill();
  fillerCSC.insert(muons, cscTimeColl.begin(), cscTimeColl.end());
  fillerCSC.fill();

  iEvent.put(std::move(muonTimeMap), "combined");
  iEvent.put(std::move(muonTimeMapDT), "dt");
  iEvent.put(std::move(muonTimeMapCSC), "csc");
}

//define this as a plug-in
//DEFINE_FWK_MODULE(MuonTimingProducer);
