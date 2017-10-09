#ifndef ZDCSIMPLERECONSTRUCTOR_H
#define ZDCSIMPLERECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParam.h" 
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/ZdcSimpleRecAlgo.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"


    /** \class HcalSimpleReconstructor	
    \author E. Garcia - CSU
    ** Based on HcalSimpleReconstructor.h by J. Mans
    */
     class ZdcSimpleReconstructor : public edm::stream::EDProducer<> {
    public:
      explicit ZdcSimpleReconstructor(const edm::ParameterSet& ps);
      virtual ~ZdcSimpleReconstructor();
      virtual void beginRun(edm::Run const&r, edm::EventSetup const & es) override final;
      virtual void endRun(edm::Run const&r, edm::EventSetup const & es) override final;
      virtual void produce(edm::Event& e, const edm::EventSetup& c) override;
    private:      
      ZdcSimpleRecAlgo reco_;
      DetId::Detector det_;
      int subdet_;
      HcalOtherSubdetector subdetOther_;
      edm::EDGetTokenT<ZDCDigiCollection> tok_input_hcal;
      edm::EDGetTokenT<ZDCDigiCollection> tok_input_castor;

      bool dropZSmarkedPassed_; // turn on/off dropping of zero suppression marked and passed digis
      
       HcalLongRecoParams* myobject; //noiseTS and signalTS from db
    };

#endif
