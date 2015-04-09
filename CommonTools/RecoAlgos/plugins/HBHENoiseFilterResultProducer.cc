// -*- C++ -*-
//
// Package:    HBHENoiseFilterResultProducer
// Class:      HBHENoiseFilterResultProducer
//
/**\class HBHENoiseFilterResultProducer

 Description: Produces the result from the HBENoiseFilter

 Implementation:
              Use the HcalNoiseSummary to make cuts on an event-by-event basis
*/
//
// Original Author:  John Paul Chou (Brown)
//
//

#include <iostream>

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/METReco/interface/HcalNoiseSummary.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

//
// class declaration
//

class HBHENoiseFilterResultProducer : public edm::EDProducer {
   public:
      explicit HBHENoiseFilterResultProducer(const edm::ParameterSet&);
      ~HBHENoiseFilterResultProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------

      // parameters
      edm::EDGetTokenT<HcalNoiseSummary> noisetoken_;
      int minHPDHits_;
      int minHPDNoOtherHits_;
      int minZeros_;

      bool IgnoreTS4TS5ifJetInLowBVRegion_;
};


//
// constructors and destructor
//

HBHENoiseFilterResultProducer::HBHENoiseFilterResultProducer(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  noisetoken_ = consumes<HcalNoiseSummary>(iConfig.getParameter<edm::InputTag>("noiselabel"));
  minHPDHits_ = iConfig.getParameter<int>("minHPDHits");
  minHPDNoOtherHits_ = iConfig.getParameter<int>("minHPDNoOtherHits");
  minZeros_ = iConfig.getParameter<int>("minZeros");
  IgnoreTS4TS5ifJetInLowBVRegion_ = iConfig.getParameter<bool>("IgnoreTS4TS5ifJetInLowBVRegion");

  produces<bool>("HBHENoiseFilterResult");
  produces<bool>("HBHENoiseFilterResultRun2Loose");
  produces<bool>("HBHENoiseFilterResultRun2Tight");
}


HBHENoiseFilterResultProducer::~HBHENoiseFilterResultProducer()
{

}


//
// member functions
//

// ------------ method called on each new Event  ------------
void
HBHENoiseFilterResultProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

  // get the Noise summary object
  edm::Handle<HcalNoiseSummary> summary_h;
  iEvent.getByToken(noisetoken_, summary_h);
  if(!summary_h.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound) << " could not find HcalNoiseSummary.\n";
    return;
  }
  const HcalNoiseSummary& summary(*summary_h);

  bool goodJetFoundInLowBVRegion = false;
  if (IgnoreTS4TS5ifJetInLowBVRegion_)
      goodJetFoundInLowBVRegion = summary.goodJetFoundInLowBVRegion();

  const bool failCommon = summary.maxHPDHits() >= minHPDHits_ ||
                          summary.maxHPDNoOtherHits() >= minHPDNoOtherHits_ ||
                          summary.maxZeros() >= minZeros_;

  const bool failRun1 = failCommon || (summary.HasBadRBXTS4TS5() &&
                                       !goodJetFoundInLowBVRegion);
  const bool failRun2Loose = failCommon || (summary.HasBadRBXRechitR45Loose() &&
                                            !goodJetFoundInLowBVRegion);
  const bool failRun2Tight = failCommon || (summary.HasBadRBXRechitR45Tight() &&
                                            !goodJetFoundInLowBVRegion);

  std::auto_ptr<bool> pOut;
  pOut = std::auto_ptr<bool>(new bool(!failRun1));
  iEvent.put(pOut, "HBHENoiseFilterResult");

  pOut = std::auto_ptr<bool>(new bool(!failRun2Loose));
  iEvent.put(pOut, "HBHENoiseFilterResultRun2Loose");

  pOut = std::auto_ptr<bool>(new bool(!failRun2Tight));
  iEvent.put(pOut, "HBHENoiseFilterResultRun2Tight");

  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HBHENoiseFilterResultProducer);
