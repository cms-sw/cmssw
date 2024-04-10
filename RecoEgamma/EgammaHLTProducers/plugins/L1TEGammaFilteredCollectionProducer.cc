/*
  Author: Swagata Mukherjee

  Date: Feb 2021

  At the time of writing this new module, it is intended to be used mainly for
  phase-2. Before feeding in the L1 e/g collection to
  HLTEcalRecHitInAllL1RegionsProducer, it can pass through this module which
  will filter the collection based on hardware quality and pT.

  The most generic L1 e/g phase-2 collections are:
  TkEm, which is std::vector<l1t::TkEm>
  &
  StaEG, which is l1t::P2GTCandidateCollection

  Despite this technical difference, the objects are almost identical, for all
  practical purposes. So any of these two collections could have been used.
  Currently, l1t::P2GTCandidateCollection is recognised by the next step
  HLTEcalRecHitInAllL1RegionsProducer, while std::vector<l1t::TkEm> is not. So
  using l1t::P2GTCandidateCollection is straightforward. If for some reason one need to
  use std::vector<l1t::TkEm>, changes in HLTEcalRecHitInAllL1RegionsProducer
  would also be necesary.
*/

#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class L1TEGammaFilteredCollectionProducer : public edm::global::EDProducer<> {
public:
  explicit L1TEGammaFilteredCollectionProducer(const edm::ParameterSet&);
  ~L1TEGammaFilteredCollectionProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  edm::InputTag l1EgTag_;
  edm::EDGetTokenT<l1t::P2GTCandidateCollection> l1EgToken_;
  int quality_;
  bool qualIsMask_;
  bool applyQual_;
  int minBX_;
  int maxBX_;
  double minPt_;
};

L1TEGammaFilteredCollectionProducer::L1TEGammaFilteredCollectionProducer(const edm::ParameterSet& iConfig)
    : l1EgTag_(iConfig.getParameter<edm::InputTag>("inputTag")),
      l1EgToken_(consumes<l1t::P2GTCandidateCollection>(l1EgTag_)) {
  quality_ = iConfig.getParameter<int>("quality");
  qualIsMask_ = iConfig.getParameter<bool>("qualIsMask");
  applyQual_ = iConfig.getParameter<bool>("applyQual");
  minBX_ = iConfig.getParameter<int>("minBX");
  maxBX_ = iConfig.getParameter<int>("maxBX");
  minPt_ = iConfig.getParameter<double>("minPt");

  produces<l1t::P2GTCandidateCollection>();
}

L1TEGammaFilteredCollectionProducer::~L1TEGammaFilteredCollectionProducer() = default;

void L1TEGammaFilteredCollectionProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag", edm::InputTag("L1EGammaClusterEmuProducer"));
  desc.add<int>("quality", 0x2);
  desc.add<bool>("qualIsMask", true);
  desc.add<bool>("applyQual", true);
  desc.add<int>("minBX", -1);
  desc.add<int>("maxBX", 1);
  desc.add<double>("minPt", 5.0);
  descriptions.add("L1TEGammaFilteredCollectionProducer", desc);
}

void L1TEGammaFilteredCollectionProducer::produce(edm::StreamID sid,
                                                  edm::Event& iEvent,
                                                  const edm::EventSetup& iSetup) const {
  auto outEgs = std::make_unique<l1t::P2GTCandidateCollection>();
  auto l1Egs = iEvent.getHandle(l1EgToken_);

  for (l1t::P2GTCandidateCollection::const_iterator iEg = (*l1Egs).begin(); iEg != (*l1Egs).end(); iEg++) {
    double offlineEt = (*iEg).pt();
    bool passQuality(false);
    if (applyQual_) {
      if (qualIsMask_)
        passQuality = ((*iEg).hwQual() & quality_);
      else
        passQuality = ((*iEg).hwQual() == quality_);
    } else
      passQuality = true;

    // if quality is passed, put the object in filtered collection
    if (passQuality && (offlineEt > minPt_)) {
      outEgs->push_back(*iEg);
    }
  }  // l1EG loop ends
  iEvent.put(std::move(outEgs));
}

DEFINE_FWK_MODULE(L1TEGammaFilteredCollectionProducer);
