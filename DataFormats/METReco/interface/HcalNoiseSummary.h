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
  bool passNoiseFilter(void) const;

  // the status with which the filter failed
  // 0 is no failure
  int noiseFilterStatus(void) const;

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
  
  // number of "problematic" RBXs
  int numProblematicRBXs(void) const;

  // reference to problematic jets
  edm::RefVector<reco::CaloJetCollection> problematicJets(void) const;

 private:

  // data members corresponding to the values above
  int filterstatus_;
  float emenergy_, hadenergy_, trackenergy_;
  float min10_, max10_, rms10_;
  float min25_, max25_, rms25_;
  int nproblemRBXs_;

  edm::RefVector<reco::CaloJetCollection> problemjets_;

};

#endif
