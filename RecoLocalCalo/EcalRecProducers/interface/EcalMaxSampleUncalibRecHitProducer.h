#ifndef RecoLocalCalo_EcalRecProducers_EcalMaxSampleUncalibRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalMaxSampleUncalibRecHitProducer_HH

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMaxSampleAlgo.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"

// forward declaration
class EcalMaxSampleUncalibRecHitProducer : public edm::EDProducer {

 public:
  typedef std::vector<double> EcalRecoAmplitudes;
  explicit EcalMaxSampleUncalibRecHitProducer(const edm::ParameterSet& ps);
  ~EcalMaxSampleUncalibRecHitProducer();
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  edm::InputTag EBdigiCollection_; // collection of EB digis
  edm::InputTag EEdigiCollection_; // collection of EE digis

  std::string EBhitCollection_; // secondary name to be given to collection of hit
  std::string EEhitCollection_; // secondary name to be given to collection of hits

  EcalUncalibRecHitMaxSampleAlgo<EBDataFrame> EBalgo_;
  EcalUncalibRecHitMaxSampleAlgo<EEDataFrame> EEalgo_;

  //    HepMatrix makeMatrixFromVectors(const std::vector< std::vector<EcalMaxSample> >& vecvec);

  /*     int nMaxPrintout_; // max # of printouts */
  /*     int counter_; // internal verbosity counter */

  //    bool counterExceeded() const { return ( (counter_>nMaxPrintout_) || (counter_<0) ) ; }
};
#endif
