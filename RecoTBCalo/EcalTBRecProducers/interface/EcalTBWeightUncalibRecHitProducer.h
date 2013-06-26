#ifndef RecoTBCalo_EcalTBRecProducers_EcalTBWeightUncalibRecHitProducer_HH
#define RecoTBCalo_EcalTBRecProducers_EcalTBWeightUncalibRecHitProducer_HH

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecWeightsAlgo.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"


// forward declaration
class EcalTBWeightUncalibRecHitProducer : public edm::EDProducer {

  public:
    typedef std::vector<double> EcalRecoAmplitudes;
    explicit EcalTBWeightUncalibRecHitProducer(const edm::ParameterSet& ps);
    ~EcalTBWeightUncalibRecHitProducer();
    virtual void produce(edm::Event& evt, const edm::EventSetup& es);

  private:

    edm::InputTag EBdigiCollection_; // secondary name given to collection of digis
    edm::InputTag EEdigiCollection_; // secondary name given to collection of digis
    edm::InputTag tdcRecInfoCollection_; // secondary name given to collection of digis

    std::string EBhitCollection_; // secondary name to be given to collection of hit
    std::string EEhitCollection_; // secondary name to be given to collection of hit

    EcalUncalibRecHitRecWeightsAlgo<EBDataFrame> EBalgo_;
    EcalUncalibRecHitRecWeightsAlgo<EEDataFrame> EEalgo_;

    const EEShape testbeamEEShape;  
    const EBShape testbeamEBShape; 

/*     HepMatrix makeMatrixFromVectors(const std::vector< std::vector<EcalWeight> >& vecvec); */
/*     HepMatrix makeDummySymMatrix(int size); */

    int nbTimeBin_;

    //use 2004 convention for the TDC
    bool use2004OffsetConvention_; 

/*     int nMaxPrintout_; // max # of printouts */
/*     int counter_; // internal verbosity counter */

    //    bool counterExceeded() const { return ( (counter_>nMaxPrintout_) || (counter_<0) ) ; }
};
#endif
