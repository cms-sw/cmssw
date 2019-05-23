#include "HLTHcalPhiSymFilter.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HLTHcalPhiSymFilter::HLTHcalPhiSymFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  HBHEHits_ = iConfig.getParameter<edm::InputTag>("HBHEHitCollection");
  HOHits_ = iConfig.getParameter<edm::InputTag>("HOHitCollection");
  HFHits_ = iConfig.getParameter<edm::InputTag>("HFHitCollection");
  phiSymHBHEHits_ = iConfig.getParameter<std::string>("phiSymHBHEHitCollection");
  phiSymHOHits_ = iConfig.getParameter<std::string>("phiSymHOHitCollection");
  phiSymHFHits_ = iConfig.getParameter<std::string>("phiSymHFHitCollection");

  eCut_HB_ = iConfig.getParameter<double>("eCut_HB");
  eCut_HE_ = iConfig.getParameter<double>("eCut_HE");
  eCut_HO_ = iConfig.getParameter<double>("eCut_HO");
  eCut_HF_ = iConfig.getParameter<double>("eCut_HF");

  HBHEHitsToken_ = consumes<HBHERecHitCollection>(HBHEHits_);
  HOHitsToken_ = consumes<HORecHitCollection>(HOHits_);
  HFHitsToken_ = consumes<HFRecHitCollection>(HFHits_);

  //register your products
  produces<HBHERecHitCollection>(phiSymHBHEHits_);
  produces<HORecHitCollection>(phiSymHOHits_);
  produces<HFRecHitCollection>(phiSymHFHits_);
}

HLTHcalPhiSymFilter::~HLTHcalPhiSymFilter() = default;

void HLTHcalPhiSymFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("HBHEHitCollection", edm::InputTag("hbhereco"));
  desc.add<edm::InputTag>("HOHitCollection", edm::InputTag("horeco"));
  desc.add<edm::InputTag>("HFHitCollection", edm::InputTag("hfreco"));
  desc.add<double>("eCut_HE", -10.);
  desc.add<double>("eCut_HF", -10.);
  desc.add<double>("eCut_HB", -10.);
  desc.add<double>("eCut_HO", -10.);
  desc.add<std::string>("phiSymHOHitCollection", "phiSymHcalRecHitsHO");
  desc.add<std::string>("phiSymHBHEHitCollection", "phiSymHcalRecHitsHBHE");
  desc.add<std::string>("phiSymHFHitCollection", "phiSymHcalRecHitsHF");
  descriptions.add("alCaHcalPhiSymStream", desc);
}

// ------------ method called to produce the data  ------------
bool HLTHcalPhiSymFilter::hltFilter(edm::Event& iEvent,
                                    const edm::EventSetup& iSetup,
                                    trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  edm::Handle<HBHERecHitCollection> HBHERecHitsH;
  edm::Handle<HORecHitCollection> HORecHitsH;
  edm::Handle<HFRecHitCollection> HFRecHitsH;

  iEvent.getByToken(HBHEHitsToken_, HBHERecHitsH);
  iEvent.getByToken(HOHitsToken_, HORecHitsH);
  iEvent.getByToken(HFHitsToken_, HFRecHitsH);

  //Create empty output collections
  std::unique_ptr<HBHERecHitCollection> phiSymHBHERecHitCollection(new HBHERecHitCollection);
  std::unique_ptr<HORecHitCollection> phiSymHORecHitCollection(new HORecHitCollection);
  std::unique_ptr<HFRecHitCollection> phiSymHFRecHitCollection(new HFRecHitCollection);

  //Select interesting HBHERecHits
  for (auto const& it : *HBHERecHitsH) {
    if (it.energy() > eCut_HB_ && it.id().subdet() == 1) {
      phiSymHBHERecHitCollection->push_back(it);
    }
    if (it.energy() > eCut_HE_ && it.id().subdet() == 2) {
      phiSymHBHERecHitCollection->push_back(it);
    }
  }

  //Select interesting HORecHits
  for (auto const& it : *HORecHitsH) {
    if (it.energy() > eCut_HO_ && it.id().subdet() == 3) {
      phiSymHORecHitCollection->push_back(it);
    }
  }

  //Select interesting HFRecHits
  for (auto const& it : *HFRecHitsH) {
    if (it.energy() > eCut_HF_ && it.id().subdet() == 4) {
      phiSymHFRecHitCollection->push_back(it);
    }
  }

  if ((phiSymHBHERecHitCollection->empty()) && (phiSymHORecHitCollection->empty()) &&
      (phiSymHFRecHitCollection->empty()))
    return false;

  //Put selected information in the event
  if (!phiSymHBHERecHitCollection->empty())
    iEvent.put(std::move(phiSymHBHERecHitCollection), phiSymHBHEHits_);
  if (!phiSymHORecHitCollection->empty())
    iEvent.put(std::move(phiSymHORecHitCollection), phiSymHOHits_);
  if (!phiSymHFRecHitCollection->empty())
    iEvent.put(std::move(phiSymHFRecHitCollection), phiSymHFHits_);

  return true;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTHcalPhiSymFilter);
