#ifndef _DATAFORMATS_METRECO_HCALNOISESUMMARY_H__
#define _DATAFORMATS_METRECO_HCALNOISESUMMARY_H__

//
// HcalNoiseSummary.h
//
//    description: Container class for HCAL noise summary information
//
//    author: J.P. Chou, Brown
//

#include "DataFormats/METReco/interface/HcalNoiseHPD.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

//
// forward declaration
//

namespace reco {
  class HcalNoiseInfoProducer;
}

//
// class definition
//

class HcalNoiseSummary {

  friend class reco::HcalNoiseInfoProducer; // allows this class to fill the info

 public:
  // constructor
  HcalNoiseSummary();

  // destructor
  virtual ~HcalNoiseSummary();

  // whether or not the event passed the event filter
  bool passLooseNoiseFilter(void) const;
  bool passTightNoiseFilter(void) const;
  bool passHighLevelNoiseFilter(void) const;

  // the status with which the filter failed
  // 0 is no failure
  int noiseFilterStatus(void) const;

  // noise type
  int noiseType(void) const;

  // quantities to calculate EM fraction and charge fraction
  // of the event (|eta|<3.0)
  float eventEMEnergy(void) const;
  float eventHadEnergy(void) const;
  float eventTrackEnergy(void) const;
  float eventEMFraction(void) const;
  float eventChargeFraction(void) const;

  // minimum/maximum/RMS rechit time
  // rechit energy>10 GeV or 25 GeV
  float min10GeVHitTime(void) const;
  float max10GeVHitTime(void) const;
  float rms10GeVHitTime(void) const;
  float min25GeVHitTime(void) const;
  float max25GeVHitTime(void) const;
  float rms25GeVHitTime(void) const;

  // # of hits with E>10 GeV or 25 GeV
  int num10GeVHits(void) const;
  int num25GeVHits(void) const;
  
  // E(2TS), E(10TS), and E(2TS)/E(10TS) for the minimum E(2TS)/E(10TS) found in an RBX in the event
  // the total energy in the RBX must be > 20 GeV
  float minE2TS(void) const;
  float minE10TS(void) const;
  float minE2Over10TS(void) const;

  // largest number of zeros found in a single RBX in the event
  int maxZeros(void) const;

  // largest number of hits in a single HPD/RBX in the event
  int maxHPDHits(void) const;
  int maxRBXHits(void) const;

  // smallest EMF found in an HPD/RBX in the event
  // the total energy in the HPD/RBX must be >20 GeV
  float minHPDEMF(void) const;
  float minRBXEMF(void) const;

  // number of "problematic" RBXs
  int numProblematicRBXs(void) const;

  // reference to problematic jets
  edm::RefVector<reco::CaloJetCollection> problematicJets(void) const;

 private:

  // data members corresponding to the values above
  int filterstatus_, noisetype_;
  float emenergy_, hadenergy_, trackenergy_;
  float min10_, max10_, rms10_;
  float min25_, max25_, rms25_;
  int cnthit10_, cnthit25_;
  float mine2ts_, mine10ts_;
  int maxzeros_;
  int maxhpdhits_, maxrbxhits_;
  float minhpdemf_, minrbxemf_;
  int nproblemRBXs_;

  edm::RefVector<reco::CaloJetCollection> problemjets_;

};

#endif
