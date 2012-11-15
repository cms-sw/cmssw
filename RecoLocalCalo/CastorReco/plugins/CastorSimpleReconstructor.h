#ifndef CASTORSIMPLERECONSTRUCTOR_H
#define CASTORSIMPLERECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/CastorReco/interface/CastorSimpleRecAlgo.h"
#include "CondFormats/CastorObjects/interface/CastorRecoParams.h"
#include "CondFormats/CastorObjects/interface/CastorSaturationCorrs.h"

class CastorSimpleReconstructor : public edm::EDProducer {
    public:
      explicit CastorSimpleReconstructor(const edm::ParameterSet& ps);
      virtual ~CastorSimpleReconstructor();
      virtual void beginRun(edm::Run&r, edm::EventSetup const & es);
      virtual void produce(edm::Event& e, const edm::EventSetup& c);
    private:      
      CastorSimpleRecAlgo reco_;
      DetId::Detector det_;
      int subdet_;
      //      HcalOtherSubdetector subdetOther_;
      edm::InputTag inputLabel_;
      
      int firstSample_;
      int samplesToAdd_;
      bool tsFromDB_;
      CastorRecoParams* paramTS_;
      bool setSaturationFlag_;
      int maxADCvalue_;
      bool doSaturationCorr_;
      CastorSaturationCorrs* satCorr_;
};

#endif
