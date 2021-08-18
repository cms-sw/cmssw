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
  StaEG, which is BXVector<l1t::EGamma>

  Despite this technical difference, the objects are almost identical, for all
  practical purposes. So any of these two collections could have been used.
  Currently, BXVector<l1t::EGamma> is recognised by the next step
  HLTEcalRecHitInAllL1RegionsProducer, while std::vector<l1t::TkEm> is not. So
  using BXVector<l1t::EGamma> is straightforward. If for some reason one need to
  use std::vector<l1t::TkEm>, changes in HLTEcalRecHitInAllL1RegionsProducer
  would also be necesary.
*/

#include "DataFormats/L1TCorrelator/interface/TkEm.h"
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
  edm::EDGetTokenT<BXVector<l1t::EGamma>> l1EgToken_;
  int quality_;
  bool qualIsMask_;
  bool applyQual_;
  int minBX_;
  int maxBX_;
  double minPt_;
  std::vector<double> scalings_;  // pT scaling factors
  double getOfflineEt(double et) const;
};

L1TEGammaFilteredCollectionProducer::L1TEGammaFilteredCollectionProducer(const edm::ParameterSet& iConfig)
    : l1EgTag_(iConfig.getParameter<edm::InputTag>("inputTag")), l1EgToken_(consumes<BXVector<l1t::EGamma>>(l1EgTag_)) {
  quality_ = iConfig.getParameter<int>("quality");
  qualIsMask_ = iConfig.getParameter<bool>("qualIsMask");
  applyQual_ = iConfig.getParameter<bool>("applyQual");
  minBX_ = iConfig.getParameter<int>("minBX");
  maxBX_ = iConfig.getParameter<int>("maxBX");
  minPt_ = iConfig.getParameter<double>("minPt");
  scalings_ = iConfig.getParameter<std::vector<double>>("scalings");

  produces<BXVector<l1t::EGamma>>();
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
  desc.add<std::vector<double>>("scalings", {2.6604, 1.06077, 0.0});
  descriptions.add("L1TEGammaFilteredCollectionProducer", desc);
}

void L1TEGammaFilteredCollectionProducer::produce(edm::StreamID sid,
                                                  edm::Event& iEvent,
                                                  const edm::EventSetup& iSetup) const {
  auto outEgs = std::make_unique<BXVector<l1t::EGamma>>();
  auto l1Egs = iEvent.getHandle(l1EgToken_);

  int startBX = std::max((*l1Egs).getFirstBX(), minBX_);
  int endBX = std::min((*l1Egs).getLastBX(), maxBX_);

  for (int bx = startBX; bx <= endBX; bx++) {
    // Loop over all L1 e/gamma objects
    for (BXVector<l1t::EGamma>::const_iterator iEg = (*l1Egs).begin(bx); iEg != (*l1Egs).end(bx); iEg++) {
      double offlineEt = this->getOfflineEt((*iEg).pt());
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
        outEgs->push_back(bx, *iEg);
      }
    }  // l1EG loop ends
  }    // BX loop ends
  iEvent.put(std::move(outEgs));
}

double L1TEGammaFilteredCollectionProducer::getOfflineEt(double et) const {
  return (scalings_.at(0) + et * scalings_.at(1) + et * et * scalings_.at(2));
}

DEFINE_FWK_MODULE(L1TEGammaFilteredCollectionProducer);
