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


//
// class declaration
//

class HBHENoiseFilterResultProducer : public edm::EDProducer {
   public:
      explicit HBHENoiseFilterResultProducer(const edm::ParameterSet&);
      ~HBHENoiseFilterResultProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      
      // ----------member data ---------------------------

      // parameters
      edm::InputTag label_;
      double minRatio_, maxRatio_;
      int minHPDHits_, minRBXHits_, minHPDNoOtherHits_;
      int minZeros_;
      double minHighEHitTime_, maxHighEHitTime_;
      double maxRBXEMF_;
      int minNumIsolatedNoiseChannels_;
      double minIsolatedNoiseSumE_, minIsolatedNoiseSumEt_;

      bool useTS4TS5_;
};


//
// constructors and destructor
//

HBHENoiseFilterResultProducer::HBHENoiseFilterResultProducer(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed
  label_ = iConfig.getParameter<edm::InputTag>("label"),
  minRatio_ = iConfig.getParameter<double>("minRatio");
  maxRatio_ = iConfig.getParameter<double>("maxRatio");
  minHPDHits_ = iConfig.getParameter<int>("minHPDHits");
  minRBXHits_ = iConfig.getParameter<int>("minRBXHits");
  minHPDNoOtherHits_ = iConfig.getParameter<int>("minHPDNoOtherHits");
  minZeros_ = iConfig.getParameter<int>("minZeros");
  minHighEHitTime_ = iConfig.getParameter<double>("minHighEHitTime");
  maxHighEHitTime_ = iConfig.getParameter<double>("maxHighEHitTime");
  maxRBXEMF_ = iConfig.getParameter<double>("maxRBXEMF");
  minNumIsolatedNoiseChannels_ = iConfig.getParameter<int>("minNumIsolatedNoiseChannels");
  minIsolatedNoiseSumE_ = iConfig.getParameter<double>("minIsolatedNoiseSumE");
  minIsolatedNoiseSumEt_ = iConfig.getParameter<double>("minIsolatedNoiseSumEt");  
  useTS4TS5_ = iConfig.getParameter<bool>("useTS4TS5");

  produces<bool>("HBHENoiseFilterResult");
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
  iEvent.getByLabel(label_, summary_h);
  if(!summary_h.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound) << " could not find HcalNoiseSummary.\n";
    return;
  }
  const HcalNoiseSummary summary = *summary_h;

  bool result=true;

  if(summary.minE2Over10TS()<minRatio_) result=false;
  if(summary.maxE2Over10TS()>maxRatio_) result=false;
  if(summary.maxHPDHits()>=minHPDHits_) result=false;
  if(summary.maxRBXHits()>=minRBXHits_) result=false;
  if(summary.maxHPDNoOtherHits()>=minHPDNoOtherHits_) result=false;
  if(summary.maxZeros()>=minZeros_) result=false;
  if(summary.min25GeVHitTime()<minHighEHitTime_) result=false;
  if(summary.max25GeVHitTime()>maxHighEHitTime_) result=false;
  if(summary.minRBXEMF()<maxRBXEMF_) result=false;
  if(summary.numIsolatedNoiseChannels()>=minNumIsolatedNoiseChannels_) result=false;
  if(summary.isolatedNoiseSumE()>=minIsolatedNoiseSumE_) result=false;
  if(summary.isolatedNoiseSumEt()>=minIsolatedNoiseSumEt_) result=false;
  if(useTS4TS5_ == true && summary.HasBadRBXTS4TS5() == true) result = false;

  std::auto_ptr<bool> pOut(new bool);
  *pOut=result;

  iEvent.put(pOut, "HBHENoiseFilterResult");
  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HBHENoiseFilterResultProducer);
