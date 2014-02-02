#ifndef RECOLOCALTRACKER_SISTRIPCLUSTERIZER_THREETHRESHOLDSTRIPCLUSTERIZER_H
#define RECOLOCALTRACKER_SISTRIPCLUSTERIZER_THREETHRESHOLDSTRIPCLUSTERIZER_H

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
class StripClusterizerAlgorithmFactory;

class OldThreeThresholdAlgorithm final : public StripClusterizerAlgorithm {

  friend class StripClusterizerAlgorithmFactory;

 public:
  
  //  void setSiStripNoiseService( SiStripNoiseService* in ){ SiStripNoiseService_=in;}
  void clusterizeDetUnit(const edm::DetSet<SiStripDigi>    & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output);
  void clusterizeDetUnit(const edmNew::DetSet<SiStripDigi> & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output);

  void initialize(const edm::EventSetup&);

  bool stripByStripBegin(uint32_t id) override {return false;}

  float channelThresholdInNoiseSigma() const { return theChannelThreshold;}
  float seedThresholdInNoiseSigma()    const { return theSeedThreshold;}
  float clusterThresholdInNoiseSigma() const { return theClusterThreshold;}

 private:

  OldThreeThresholdAlgorithm(float strip_thr, float seed_thr,float clust_thr, int max_holes,std::string qualityLabel,bool setDetId) :
    theChannelThreshold(strip_thr), 
    theSeedThreshold(seed_thr),
    theClusterThreshold(clust_thr),
    max_holes_(max_holes),
    qualityLabel_(qualityLabel){
    _setDetId=setDetId;};

  //  SiStripNoiseService* SiStripNoiseService_; 
  template<typename InputDetSet>
    void clusterizeDetUnit_(const InputDetSet & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output);

  float theChannelThreshold;
  float theSeedThreshold;
  float theClusterThreshold;
  int max_holes_;
  std::string qualityLabel_;

  edm::ESHandle<SiStripGain> gainHandle_;
  edm::ESHandle<SiStripNoises> noiseHandle_;
  edm::ESHandle<SiStripQuality> qualityHandle_;

  std::vector<SiStripDigi> cluster_digis_;  // so it's not recreated for each det for each event!
};

class AboveSeed {
 public:


  //  AboveSeed(float aseed,SiStripNoiseService* noise,const uint32_t& detID) : seed(aseed), noise_(noise),detID_(detID) {};

  AboveSeed(float aseed, const edm::ESHandle<SiStripNoises> & noiseHandle, const SiStripNoises::Range & noiseRange, const edm::ESHandle<SiStripQuality> & qualityHandle, const SiStripQuality::Range & qualityRange) 
    : seed(aseed), noise_(noiseHandle), noiseRange_(noiseRange),quality_(qualityHandle), qualityRange_(qualityRange)
    {};
  //,detID_(detID) {};

  inline bool operator()(const SiStripDigi& digi) { 
    return ( 
	    !quality_->IsStripBad(qualityRange_,digi.strip()) 
	        && 
	    digi.adc() >= seed * noise_->getNoise(digi.strip(), noiseRange_)
	    );
  }
 private:
  float seed;
  const edm::ESHandle<SiStripNoises> & noise_;
  const SiStripNoises::Range & noiseRange_;
  const edm::ESHandle<SiStripQuality> & quality_;
  const SiStripQuality::Range & qualityRange_;
};

#endif
