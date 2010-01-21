#ifndef ZDCSIMPLERECONSTRUCTOR_H
#define ZDCSIMPLERECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/ZdcSimpleRecAlgo.h"


    /** \class HcalSimpleReconstructor	
     $Date: 2010/01/18 00:00:66 $
    $Revision: 1.6 $
    \author E. Garcia - CSU
    ** Based on HcalSimpleReconstructor.h by J. Mans
    */
     class ZdcSimpleReconstructor : public edm::EDProducer {
    public:
      explicit ZdcSimpleReconstructor(const edm::ParameterSet& ps);
      virtual ~ZdcSimpleReconstructor();
      virtual void produce(edm::Event& e, const edm::EventSetup& c);
    private:      
      ZdcSimpleRecAlgo reco_;
      DetId::Detector det_;
      int subdet_;
      HcalOtherSubdetector subdetOther_;
      edm::InputTag inputLabel_;

      bool dropZSmarkedPassed_; // turn on/off dropping of zero suppression marked and passed digis
    };

#endif
