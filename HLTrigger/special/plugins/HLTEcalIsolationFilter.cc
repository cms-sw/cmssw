#include "HLTEcalIsolationFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

HLTEcalIsolationFilter::HLTEcalIsolationFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  candTag_ = iConfig.getParameter<edm::InputTag>("EcalIsolatedParticleSource");
  maxhitout = iConfig.getParameter<int>("MaxNhitOuterCone");
  maxhitin = iConfig.getParameter<int>("MaxNhitInnerCone");
  maxenin = iConfig.getParameter<double>("MaxEnergyInnerCone");
  maxenout = iConfig.getParameter<double>("MaxEnergyOuterCone");
  maxetacand = iConfig.getParameter<double>("MaxEtaCandidate");
  candToken_ = consumes<reco::IsolatedPixelTrackCandidateCollection>(candTag_);
}

HLTEcalIsolationFilter::~HLTEcalIsolationFilter() = default;

void HLTEcalIsolationFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("EcalIsolatedParticleSource", edm::InputTag("ecalIsolPartProd"));
  desc.add<int>("MaxNhitInnerCone", 1000);
  desc.add<int>("MaxNhitOuterCone", 0);
  desc.add<double>("MaxEnergyOuterCone", 10000.);
  desc.add<double>("MaxEnergyInnerCone", 10000.);
  desc.add<double>("MaxEtaCandidate", 1.3);
  descriptions.add("hltEcalIsolationFilter", desc);
}

bool HLTEcalIsolationFilter::hltFilter(edm::Event& iEvent,
                                       const edm::EventSetup& iSetup,
                                       trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref;

  // get hold of filtered candidates
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> ecIsolCands;
  iEvent.getByToken(candToken_, ecIsolCands);

  //Filtering

  unsigned int n = 0;
  for (unsigned int i = 0; i < ecIsolCands->size(); i++) {
    candref = edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(ecIsolCands, i);

    if ((candref->nHitIn() <= maxhitin) && (candref->nHitOut() <= maxhitout) && (candref->energyOut() < maxenout) &&
        (candref->energyIn() < maxenin) && fabs(candref->eta()) < maxetacand) {
      filterproduct.addObject(trigger::TriggerTrack, candref);
      n++;
    }
  }

  bool accept(n > 0);

  return accept;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTEcalIsolationFilter);
