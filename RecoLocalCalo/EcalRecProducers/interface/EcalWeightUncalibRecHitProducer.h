#ifndef RecoLocalCalo_EcalRecProducers_EcalWeightUncalibRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalWeightUncalibRecHitProducer_HH

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "TestEcal/Algo/src/AMPRecoWeights.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecWeightsAlgo.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "CondFormats/EcalObjects/interface/EcalWeight.h"

using namespace cms;

// forward declaration
class EcalWeightUncalibRecHitProducer : public edm::EDProducer {

  public:
    typedef std::vector<double> EcalRecoAmplitudes;
    explicit EcalWeightUncalibRecHitProducer(const edm::ParameterSet& ps);
    ~EcalWeightUncalibRecHitProducer();
    virtual void produce(edm::Event& evt, const edm::EventSetup& es);

  private:
    std::string digiProducer_;
    std::string digiCollection_;
    std::string hitCollection_;

    EcalUncalibRecHitRecWeightsAlgo<EBDataFrame> algo_;
    HepMatrix makeMatrixFromVectors(const std::vector< std::vector<EcalWeight> >& vecvec);
};
#endif
