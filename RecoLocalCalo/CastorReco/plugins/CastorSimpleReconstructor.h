#ifndef CASTORSIMPLERECONSTRUCTOR_H
#define CASTORSIMPLERECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/CastorReco/interface/CastorSimpleRecAlgo.h"

class CastorSimpleReconstructor : public edm::EDProducer {
    public:
      explicit CastorSimpleReconstructor(const edm::ParameterSet& ps);
      virtual ~CastorSimpleReconstructor();
      virtual void produce(edm::Event& e, const edm::EventSetup& c);
    private:      
      CastorSimpleRecAlgo reco_;
      DetId::Detector det_;
      int subdet_;
      //      HcalOtherSubdetector subdetOther_;
      edm::InputTag inputLabel_;
};

#endif
