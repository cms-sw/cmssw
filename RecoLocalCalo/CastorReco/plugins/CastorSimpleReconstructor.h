#ifndef CASTORSIMPLERECONSTRUCTOR_H
#define CASTORSIMPLERECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/CastorReco/interface/CastorSimpleRecAlgo.h"

class CastorSimpleReconstructor : public edm::stream::EDProducer<> {
    public:
      explicit CastorSimpleReconstructor(const edm::ParameterSet& ps);
      virtual ~CastorSimpleReconstructor();
      virtual void produce(edm::Event& e, const edm::EventSetup& c);
    private:      
      CastorSimpleRecAlgo reco_;
      DetId::Detector det_;
      int subdet_;
      //      HcalOtherSubdetector subdetOther_;
      edm::EDGetTokenT<CastorDigiCollection> tok_input_;
      
      int firstSample_;
      int samplesToAdd_;
      int maxADCvalue_;
      bool tsFromDB_;
      bool setSaturationFlag_;
      bool doSaturationCorr_;
};

#endif
