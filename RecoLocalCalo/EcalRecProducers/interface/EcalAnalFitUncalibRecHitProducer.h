#ifndef RecoLocalCalo_EcalRecProducers_EcalAnalFitUncalibRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalAnalFitUncalibRecHitProducer_HH

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAnalFitAlgo.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"

// forward declaration
class EcalAnalFitUncalibRecHitProducer : public edm::EDProducer {

  public:
    typedef std::vector<double> EcalRecoAmplitudes;
    explicit EcalAnalFitUncalibRecHitProducer(const edm::ParameterSet& ps);
    ~EcalAnalFitUncalibRecHitProducer();
    virtual void produce(edm::Event& evt, const edm::EventSetup& es);

  private:
    std::string digiProducer_; // name of module/plugin/producer making digis
    std::string EBdigiCollection_; // secondary name given to collection of digis
    std::string EEdigiCollection_; // secondary name given to collection of digis
    std::string EBhitCollection_; // secondary name to be given to collection of hit
    std::string EEhitCollection_; // secondary name to be given to collection of hits

    EcalUncalibRecHitRecAnalFitAlgo<EBDataFrame> EBalgo_;
    EcalUncalibRecHitRecAnalFitAlgo<EEDataFrame> EEalgo_;

/*     int nMaxPrintout_; // max # of printouts */
/*     int nEvt_; // internal counter of events */

/*     bool counterExceeded() const { return ( (nEvt_>nMaxPrintout_) || (nMaxPrintout_<0) ) ; } */
};
#endif
