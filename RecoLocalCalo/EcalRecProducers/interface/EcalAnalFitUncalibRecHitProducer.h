#ifndef RecoLocalCalo_EcalRecProducers_EcalAnalFitUncalibRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalAnalFitUncalibRecHitProducer_HH

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAnalFitAlgo.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"

// forward declaration
class EcalAnalFitUncalibRecHitProducer : public edm::EDProducer {

  public:
    typedef std::vector<double> EcalRecoAmplitudes;
    explicit EcalAnalFitUncalibRecHitProducer(const edm::ParameterSet& ps);
    ~EcalAnalFitUncalibRecHitProducer();
    virtual void produce(edm::Event& evt, const edm::EventSetup& es);

  private:
    std::string digiProducer_; // name of module/plugin/producer making digis
    std::string digiCollection_; // secondary name given to collection of digis
    std::string hitCollection_; // secondary name to be given to collection of hits

    EcalUncalibRecHitRecAnalFitAlgo<EBDataFrame> algo_;

    int nMaxPrintout_; // max # of printouts
    int nEvt_; // internal counter of events

    bool counterExceeded() const { return ( (nEvt_>nMaxPrintout_) || (nMaxPrintout_<0) ) ; }
};
#endif
