#ifndef RecoLocalCalo_EcalRecProducers_EcalFixedAlphaBetaFitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalFixedAlphaBetaFitProducer_HH

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitFixedAlphaBetaAlgo.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"


// forward declaration
class EcalFixedAlphaBetaFitUncalibRecHitProducer : public edm::EDProducer {

  public:
    typedef std::vector<double> EcalRecoAmplitudes;
    explicit EcalFixedAlphaBetaFitUncalibRecHitProducer(const edm::ParameterSet& ps);
    ~EcalFixedAlphaBetaFitUncalibRecHitProducer();
    virtual void produce(edm::Event& evt, const edm::EventSetup& es);

  private:

    edm::InputTag EBdigiCollection_; // secondary name given to collection of digis
    edm::InputTag EEdigiCollection_; // secondary name given to collection of digis
    std::string EBhitCollection_; // secondary name to be given to collection of hit
    std::string EEhitCollection_; // secondary name to be given to collection of hits

    double AmplThrEB_;
    double AmplThrEE_;

   EcalUncalibRecHitFixedAlphaBetaAlgo<EBDataFrame> algoEB_;
   EcalUncalibRecHitFixedAlphaBetaAlgo<EEDataFrame> algoEE_;

    double alphaEB_;
    double betaEB_;
    double alphaEE_;
    double betaEE_;
    std::vector<std::vector<std::pair<double,double> > > alphaBetaValues_;//List of alpha and Beta values [SM#][CRY#](alpha, beta)
    bool useAlphaBetaArray_;
    std::string alphabetaFilename_;
    
    bool setAlphaBeta();//Sets the alphaBetaValues_ vectors by the values provided in alphabetaFilename_

};
#endif
