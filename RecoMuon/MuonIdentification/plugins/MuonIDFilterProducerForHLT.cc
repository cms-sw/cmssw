/** \class MuonIDFilterProducerForHLT
 *
 *  \author S. Folgueras <santiago.folgueras@cern.ch>
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "RecoMuon/MuonIdentification/plugins/MuonIDFilterProducerForHLT.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

//#include <algorithm>

MuonIDFilterProducerForHLT::MuonIDFilterProducerForHLT(const edm::ParameterSet& iConfig)
    : muonTag_(iConfig.getParameter<edm::InputTag>("inputMuonCollection")),
      muonToken_(consumes<reco::MuonCollection>(muonTag_)),
      applyTriggerIdLoose_(iConfig.getParameter<bool>("applyTriggerIdLoose")),
      type_(muon::SelectionType(iConfig.getParameter<unsigned int>("typeMuon"))),
      allowedTypeMask_(iConfig.getParameter<unsigned int>("allowedTypeMask")),
      requiredTypeMask_(iConfig.getParameter<unsigned int>("requiredTypeMask")),
      min_NMuonHits_(iConfig.getParameter<int>("minNMuonHits")),
      min_NMuonStations_(iConfig.getParameter<int>("minNMuonStations")),
      min_NTrkLayers_(iConfig.getParameter<int>("minNTrkLayers")),
      min_NTrkHits_(iConfig.getParameter<int>("minTrkHits")),
      min_PixLayers_(iConfig.getParameter<int>("minPixLayer")),
      min_PixHits_(iConfig.getParameter<int>("minPixHits")),
      min_Pt_(iConfig.getParameter<double>("minPt")),
      max_NormalizedChi2_(iConfig.getParameter<double>("maxNormalizedChi2")) {
  produces<reco::MuonCollection>();
}

MuonIDFilterProducerForHLT::~MuonIDFilterProducerForHLT() {}
void MuonIDFilterProducerForHLT::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputMuonCollection", edm::InputTag("hltIterL3MuonsNoID"));
  desc.add<bool>("applyTriggerIdLoose", true);
  desc.add<unsigned int>("typeMuon", 0);
  desc.add<unsigned int>("allowedTypeMask", 0);
  desc.add<unsigned int>("requiredTypeMask", 0);
  desc.add<int>("minNMuonHits", 0);
  desc.add<int>("minNMuonStations", 0);
  desc.add<int>("minNTrkLayers", 0);
  desc.add<int>("minTrkHits", 0);
  desc.add<int>("minPixLayer", 0);
  desc.add<int>("minPixHits", 0);
  desc.add<double>("minPt", 0.);
  desc.add<double>("maxNormalizedChi2", 9999.);
  descriptions.addWithDefaultLabel(desc);
}
void MuonIDFilterProducerForHLT::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto output = std::make_unique<reco::MuonCollection>();

  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(muonToken_, muons);

  for (unsigned int i = 0; i < muons->size(); ++i) {
    const reco::Muon& muon(muons->at(i));
    if (applyTriggerIdLoose_ && muon::isLooseTriggerMuon(muon)) {
      output->push_back(muon);
    } else {  // Implement here manually all the required/desired cuts
      if ((muon.type() & allowedTypeMask_) == 0)
        continue;
      if ((muon.type() & requiredTypeMask_) != requiredTypeMask_)
        continue;
      // tracker cuts
      if (!muon.innerTrack().isNull()) {
        if (muon.innerTrack()->hitPattern().trackerLayersWithMeasurement() < min_NTrkLayers_)
          continue;
        if (muon.innerTrack()->numberOfValidHits() < min_NTrkHits_)
          continue;
        if (muon.innerTrack()->hitPattern().pixelLayersWithMeasurement() < min_PixLayers_)
          continue;
        if (muon.innerTrack()->hitPattern().numberOfValidPixelHits() < min_PixHits_)
          continue;
      }
      // muon cuts
      if (muon.numberOfMatchedStations() < min_NMuonStations_)
        continue;
      if (!muon.globalTrack().isNull()) {
        if (muon.globalTrack()->normalizedChi2() > max_NormalizedChi2_)
          continue;
        if (muon.globalTrack()->hitPattern().numberOfValidMuonHits() < min_NMuonHits_)
          continue;
      }
      if (!muon::isGoodMuon(muon, type_))
        continue;
      if (muon.pt() < min_Pt_)
        continue;

      output->push_back(muon);
    }
  }

  iEvent.put(std::move(output));
}
