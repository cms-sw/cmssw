#ifndef RecoLocalCalo_EcalRecProducers_EcalWeightUncalibRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalWeightUncalibRecHitProducer_HH

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecWeightsAlgo.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "CondFormats/EcalObjects/interface/EcalWeight.h"

// forward declaration
class EcalWeightUncalibRecHitProducer : public edm::EDProducer {

  public:
    typedef std::vector<double> EcalRecoAmplitudes;
    explicit EcalWeightUncalibRecHitProducer(const edm::ParameterSet& ps);
    ~EcalWeightUncalibRecHitProducer();
    virtual void produce(edm::Event& evt, const edm::EventSetup& es);

  private:
    std::string digiProducer_; // name of module/plugin/producer making digis
    std::string digiCollection_; // secondary name given to collection of digis
    std::string hitCollection_; // secondary name to be given to collection of hits

    EcalUncalibRecHitRecWeightsAlgo<EBDataFrame> algo_;
    HepMatrix makeMatrixFromVectors(const std::vector< std::vector<EcalWeight> >& vecvec);

    int nMaxPrintout_; // max # of printouts
    int nEvt_; // internal counter of events

    bool counterExceeded() const { return ( (nEvt_>nMaxPrintout_) || (nMaxPrintout_<0) ) ; }
};
#endif
