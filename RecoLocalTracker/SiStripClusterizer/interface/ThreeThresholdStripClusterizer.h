#ifndef RECOLOCALTRACKER_SISTRIPCLUSTERIZER_THREETHRESHOLDSTRIPCLUSTERIZER_H
#define RECOLOCALTRACKER_SISTRIPCLUSTERIZER_THREETHRESHOLDSTRIPCLUSTERIZER_H

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//SiStripNoiseService
//#include "CommonTools/SiStripZeroSuppression/interface/SiStripNoiseService.h"

//gain
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"


#include <vector>
#include <algorithm>
#include <cmath>


class ThreeThresholdStripClusterizer {
public:

  ThreeThresholdStripClusterizer(float strip_thr, float seed_thr,float clust_thr, int max_holes) :
    theChannelThreshold(strip_thr), 
    theSeedThreshold(seed_thr),
    theClusterThreshold(clust_thr),
    max_holes_(max_holes){};
 
  //  void setSiStripNoiseService( SiStripNoiseService* in ){ SiStripNoiseService_=in;}

  void clusterizeDetUnit(const 
edm::DetSet<SiStripDigi>&,edm::DetSet<SiStripCluster>&, const edm::ESHandle<SiStripNoises> & noiseHandle, const edm::ESHandle<SiStripGain>&, const 
edm::ESHandle<SiStripQuality>&);

  float channelThresholdInNoiseSigma() const { return theChannelThreshold;}
  float seedThresholdInNoiseSigma()    const { return theSeedThreshold;}
  float clusterThresholdInNoiseSigma() const { return theClusterThreshold;}

private:
  //  SiStripNoiseService* SiStripNoiseService_; 

  float theChannelThreshold;
  float theSeedThreshold;
  float theClusterThreshold;
  int max_holes_;

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
	    !noise_->getDisable(digi.strip(), noiseRange_) 
	    && 
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
