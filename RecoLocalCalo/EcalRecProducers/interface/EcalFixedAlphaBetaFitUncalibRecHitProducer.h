#ifndef RecoLocalCalo_EcalRecProducers_EcalFixedAlphaBetaFitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalFixedAlphaBetaFitProducer_HH

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitFixedAlphaBetaAlgo.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "CondFormats/EcalObjects/interface/EcalWeight.h"

// forward declaration
class EcalFixedAlphaBetaFitUncalibRecHitProducer : public edm::EDProducer {

  public:
    typedef std::vector<double> EcalRecoAmplitudes;
    explicit EcalFixedAlphaBetaFitUncalibRecHitProducer(const edm::ParameterSet& ps);
    ~EcalFixedAlphaBetaFitUncalibRecHitProducer();
    virtual void produce(edm::Event& evt, const edm::EventSetup& es);

  private:

    std::string digiProducer_; // name of module/plugin/producer making digis
    std::string EBdigiCollection_; // secondary name given to collection of digis
    std::string EEdigiCollection_; // secondary name given to collection of digis
    std::string EBhitCollection_; // secondary name to be given to collection of hit
    std::string EEhitCollection_; // secondary name to be given to collection of hits


   EcalUncalibRecHitFixedAlphaBetaAlgo<EBDataFrame> algoEB_;
   EcalUncalibRecHitFixedAlphaBetaAlgo<EEDataFrame> algoEE_;
};
#endif
