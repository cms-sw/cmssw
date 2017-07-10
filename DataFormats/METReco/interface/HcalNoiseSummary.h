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
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

//
// forward declaration
//

namespace reco {
  class HcalNoiseInfoProducer;
}

//
// class definition
//

class HcalNoiseSummary
{
  friend class reco::HcalNoiseInfoProducer; // allows this class to fill the info

 public:
  // constructor
  HcalNoiseSummary();

  // destructor
  virtual ~HcalNoiseSummary();

  // whether or not the event passed the event filter
  // note that these methods are deprecated
  // please see the instructions here: https://twiki.cern.ch/twiki/bin/view/CMS/HcalNoiseInfoLibrary
  bool passLooseNoiseFilter() const;
  bool passTightNoiseFilter() const;
  bool passHighLevelNoiseFilter() const;

  // the status with which the filter failed: this is a bitset
  // 0 is no failure
  int noiseFilterStatus() const;

  // noise type: 1=HPD Ionfeedback, 2=HPD Discharge, 3=RBX Noise
  // won't work with non-noise event
  int noiseType() const;

  // quantities to calculate EM fraction and charge fraction
  // of the event (|eta|<2.0)
  float eventEMEnergy() const;
  float eventHadEnergy() const;
  float eventTrackEnergy() const;
  float eventEMFraction() const;
  float eventChargeFraction() const;

  // minimum/maximum/RMS rechit time
  // rechit energy>10 GeV or 25 GeV
  float min10GeVHitTime() const;
  float max10GeVHitTime() const;
  float rms10GeVHitTime() const;
  float min25GeVHitTime() const;
  float max25GeVHitTime() const;
  float rms25GeVHitTime() const;

  // # of hits with E>10 GeV or 25 GeV
  int num10GeVHits() const;
  int num25GeVHits() const;
  
  // E(2TS), E(10TS), and E(2TS)/E(10TS) for the minimum and maximum E(2TS)/E(10TS) found in an RBX in the event
  // the total energy in the RBX must be > 50 GeV
  float minE2TS() const;
  float minE10TS() const;
  float minE2Over10TS() const;
  float maxE2TS() const;
  float maxE10TS() const;
  float maxE2Over10TS() const;

  // largest number of zeros found in a single RBX in the event
  // total energy in the RBX must be > 10 GeV
  int maxZeros() const;

  // largest number of hits in a single HPD/RBX in the event
  // each hit is >= 1.5 GeV
  int maxHPDHits() const;
  int maxRBXHits() const;

  // largest number of hits in a single HPD when no other hits are present in the RBX
  int maxHPDNoOtherHits() const;

  // smallest EMF found in an HPD/RBX in the event
  // the total energy in the HPD/RBX must be >50 GeV
  float minHPDEMF() const;
  float minRBXEMF() const;

  // number of "problematic" RBXs
  int numProblematicRBXs() const;

  int numIsolatedNoiseChannels() const;
  float isolatedNoiseSumE() const;
  float isolatedNoiseSumEt() const;

  int numFlatNoiseChannels() const;
  float flatNoiseSumE() const;
  float flatNoiseSumEt() const;

  int numSpikeNoiseChannels() const;
  float spikeNoiseSumE() const;
  float spikeNoiseSumEt() const;

  int numTriangleNoiseChannels() const;
  float triangleNoiseSumE() const;
  float triangleNoiseSumEt() const;

  int numTS4TS5NoiseChannels() const;
  float TS4TS5NoiseSumE() const;
  float TS4TS5NoiseSumEt() const;

  int numNegativeNoiseChannels() const;
  float NegativeNoiseSumE() const;
  float NegativeNoiseSumEt() const;

  int GetRecHitCount() const;
  int GetRecHitCount15() const;
  double GetRecHitEnergy() const;
  double GetRecHitEnergy15() const;

  double GetTotalCalibCharge() const;

  bool HasBadRBXTS4TS5() const;
  bool HasBadRBXRechitR45Loose() const;
  bool HasBadRBXRechitR45Tight() const;
  bool goodJetFoundInLowBVRegion() const;

  double GetCalibChargeHF() const;
  int    GetCalibCountHF()  const;

  // Get charge only in TS45
  int GetCalibCountTS45() const;  // get number of HBHE calibration channels
  int GetCalibgt15CountTS45() const; // get number of HBHE calib channels > 15 fC
  double GetCalibChargeTS45() const; // get Calib charge
  double GetCalibgt15ChargeTS45() const; // get charge from all channels gt 15 fC

  int GetHitsInNonLaserRegion() const; // get number of channels in HBHE regions with no laser
  int GetHitsInLaserRegion() const; // get number of channels in HBHE region where laser pulses are seen
  double GetEnergyInNonLaserRegion() const; // get energy in region with no laser
  double GetEnergyInLaserRegion() const; // get energy in non-laser region
  
  // reference to problematic jets
  edm::RefVector<reco::CaloJetCollection> problematicJets() const;

  // reference to calotowers which fail loose, tight, and high-level noise criteria
  edm::RefVector<CaloTowerCollection> looseNoiseTowers() const;
  edm::RefVector<CaloTowerCollection> tightNoiseTowers() const;
  edm::RefVector<CaloTowerCollection> highLevelNoiseTowers() const;

 private:

  // data members corresponding to the values above
  int filterstatus_, noisetype_;
  float emenergy_, hadenergy_, trackenergy_;
  float min10_, max10_, rms10_;
  float min25_, max25_, rms25_;
  int cnthit10_, cnthit25_;
  float mine2ts_, mine10ts_;
  float maxe2ts_, maxe10ts_;
  int maxzeros_;
  int maxhpdhits_, maxhpdhitsnoother_, maxrbxhits_;
  float minhpdemf_, minrbxemf_;
  int nproblemRBXs_;
  int nisolnoise_;
  float isolnoisee_, isolnoiseet_;
  int nflatnoise_;
  float flatnoisee_, flatnoiseet_;
  int nspikenoise_;
  float spikenoisee_, spikenoiseet_;
  int ntrianglenoise_;
  float trianglenoisee_, trianglenoiseet_;
  int nts4ts5noise_;
  float ts4ts5noisee_, ts4ts5noiseet_;
  int nnegativenoise_;
  float negativenoisee_, negativenoiseet_;

  int rechitCount_;
  int rechitCount15_;
  double rechitEnergy_;
  double rechitEnergy15_;
  double calibCharge_;

  bool hasBadRBXTS4TS5_;
  bool hasBadRBXRechitR45Loose_;
  bool hasBadRBXRechitR45Tight_;
  bool goodJetFoundInLowBVRegion_;

  int calibCountTS45_;
  int calibCountgt15TS45_;
  double calibChargeTS45_;
  double calibChargegt15TS45_;

  int calibCountHF_; // calibration channels only in HF; no threshold used for determining HF noise
  double calibChargeHF_;

  int hitsInLaserRegion_;
  int hitsInNonLaserRegion_;
  double energyInLaserRegion_;
  double energyInNonLaserRegion_;

  edm::RefVector<reco::CaloJetCollection> problemjets_;

  edm::RefVector<CaloTowerCollection> loosenoisetwrs_;
  edm::RefVector<CaloTowerCollection> tightnoisetwrs_;
  edm::RefVector<CaloTowerCollection> hlnoisetwrs_;
};

#endif
