#ifndef ZDCSIMPLERECONSTRUCTOR_H
#define ZDCSIMPLERECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalLongRecoParam.h" 
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/ZdcSimpleRecAlgo.h"


    /** \class HcalSimpleReconstructor	
     $Date: 2011/02/23 22:51:18 $
    $Revision: 1.2 $
    \author E. Garcia - CSU
    ** Based on HcalSimpleReconstructor.h by J. Mans
    */
     class ZdcSimpleReconstructor : public edm::EDProducer {
    public:
      explicit ZdcSimpleReconstructor(const edm::ParameterSet& ps);
      virtual ~ZdcSimpleReconstructor();
      virtual void beginRun(edm::Run&r, edm::EventSetup const & es);
      virtual void endRun(edm::Run&r, edm::EventSetup const & es);
      virtual void produce(edm::Event& e, const edm::EventSetup& c);
    private:      
      ZdcSimpleRecAlgo reco_;
      DetId::Detector det_;
      int subdet_;
      HcalOtherSubdetector subdetOther_;
      edm::InputTag inputLabel_;

      bool dropZSmarkedPassed_; // turn on/off dropping of zero suppression marked and passed digis
      
       HcalLongRecoParams* myobject; //noiseTS and signalTS from db
    };

#endif
