// -*- C++ -*-
//
// Package:    EcalSimpleUncalibRecHitFilter
// Class:      EcalSimpleUncalibRecHitFilter
//
/**\class EcalSimpleUncalibRecHitFilter EcalSimpleUncalibRecHitFilter.cc Work/EcalSimpleUncalibRecHitFilter/src/EcalSimpleUncalibRecHitFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Giovanni FRANZONI
//         Created:  Wed Sep 19 16:21:29 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//
// class declaration
//

class EcalSimpleUncalibRecHitFilter : public edm::EDFilter {
public:
  explicit EcalSimpleUncalibRecHitFilter(const edm::ParameterSet &);
  ~EcalSimpleUncalibRecHitFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  bool filter(edm::Event &, edm::EventSetup const &) override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<EcalUncalibratedRecHitCollection> EcalUncalibRecHitToken_;
  const double minAdc_;
  const std::vector<int> maskedList_;
};

//
// constructors and destructor
//
EcalSimpleUncalibRecHitFilter::EcalSimpleUncalibRecHitFilter(const edm::ParameterSet &iConfig)
    : EcalUncalibRecHitToken_(consumes<EcalUncalibratedRecHitCollection>(
          iConfig.getParameter<edm::InputTag>("EcalUncalibRecHitCollection"))),
      minAdc_(iConfig.getUntrackedParameter<double>("adcCut", 12)),
      maskedList_(iConfig.getUntrackedParameter<std::vector<int>>("maskedChannels",
                                                                  std::vector<int>{}))  // this is using the ashed index
{
  // now do what ever initialization is needed
}

EcalSimpleUncalibRecHitFilter::~EcalSimpleUncalibRecHitFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool EcalSimpleUncalibRecHitFilter::filter(edm::Event &iEvent, edm::EventSetup const &iSetup) {
  using namespace edm;

  // getting very basic uncalRH
  Handle<EcalUncalibratedRecHitCollection> crudeHits;
  if (not iEvent.getByToken(EcalUncalibRecHitToken_, crudeHits)) {
    edm::EDConsumerBase::Labels labels;
    labelsForToken(EcalUncalibRecHitToken_, labels);
    LogWarning("EcalSimpleUncalibRecHitFilter")
        << "InputTag:  label = \"" << labels.module << "\", instance = \"" << labels.productInstance
        << "\", process = \"" << labels.process << "\" is not available";
    return false;
  }

  bool thereIsSignal = false;
  // loop on crude rechits
  for (auto hit : *crudeHits) {
    // masking noisy channels
    auto result = std::find(maskedList_.begin(), maskedList_.end(), EBDetId(hit.id()).hashedIndex());
    if (result != maskedList_.end())
      // LogWarning("EcalFilter") << "skipping uncalRecHit for channel: " << ic << " with amplitude " << ampli_ ;
      continue;

    float ampli_ = hit.amplitude();

    // seeking channels with signal and displaced jitter
    if (ampli_ >= minAdc_) {
      thereIsSignal = true;
      // LogWarning("EcalFilter")  << "at evet: " << iEvent.id().event()
      // 				       << " and run: " << iEvent.id().run()
      // 				       << " there is OUT OF TIME signal at chanel: " << ic
      // 				       << " with amplitude " << ampli_  << " and max at: " << jitter_;
      break;
    }
  }

  return thereIsSignal;
}

void EcalSimpleUncalibRecHitFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("EcalUncalibRecHitCollection",
                          edm::InputTag("ecalWeightUncalibRecHit", "EcalUncalibRecHitsEB"));
  desc.addUntracked<double>("adcCut", 12.);
  desc.addUntracked<std::vector<int>>("maskedChannels", std::vector<int>{});

  descriptions.add("ecalSimpleUncalibRecHitFilter", desc);
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalSimpleUncalibRecHitFilter);
