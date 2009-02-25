#ifndef ANALYSISDATAFORMATS_SISTRIPCLUSTERINFO_H
#define ANALYSISDATAFORMATS_SISTRIPCLUSTERINFO_H

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include "boost/cstdint.hpp"

class SiStripClusterInfo {
 public:

  SiStripClusterInfo(const SiStripCluster& cluster, 
		     const edm::EventSetup& es, 
		     std::string qualityLabel="");
  ~SiStripClusterInfo() {}

  const SiStripCluster * cluster() const {return cluster_;}

  uint32_t detId() const      {return cluster()->geographicalId();}
  uint16_t width() const      {return cluster()->amplitudes().size();}
  uint16_t firstStrip() const {return cluster()->firstStrip();}
  float    baryStrip() const  {return cluster()->barycenter();}
  uint16_t maxStrip() const   {return firstStrip() + maxIndex();}

  const std::vector<uint8_t>& stripCharges() const {return cluster()->amplitudes();}
  std::vector<float>          stripGains() const;
  std::vector<float>          stripNoises() const;
  std::vector<float>          stripNoisesRescaledByGain() const;
  std::vector<bool>           stripQualitiesBad() const;

  uint16_t charge() const    {return   accumulate( stripCharges().begin(), stripCharges().end(), uint16_t(0));}
  uint8_t  maxCharge() const {return * max_element(stripCharges().begin(), stripCharges().end());}
  uint16_t maxIndex() const  {return   max_element(stripCharges().begin(), stripCharges().end()) - stripCharges().begin();}
  std::pair<uint16_t,uint16_t> chargeLR() const;
  
  float noise() const               { return calculate_noise(stripNoises());}
  float noiseRescaledByGain() const { return calculate_noise(stripNoisesRescaledByGain());}

  float signalOverNoise() const { return charge()/noiseRescaledByGain(); }

  bool IsAnythingBad() const;
  bool IsApvBad() const;
  bool IsFiberBad() const;
  bool IsModuleBad() const;
  bool IsModuleUsable() const;

  std::vector<SiStripCluster> reclusterize(float channelThreshold, 
					   float seedThreshold, 
					   float clusterThreshold, 
					   uint8_t maxSequentialHoles,
					   uint8_t maxSequentialBad,
					   uint8_t maxAdjacentBad) const;

 private:
  float calculate_noise(const std::vector<float>&) const; 

  const SiStripCluster* cluster_;
  const edm::EventSetup& es_; 
  edm::ESHandle<SiStripNoises> noiseHandle_;
  edm::ESHandle<SiStripGain> gainHandle_;
  edm::ESHandle<SiStripQuality> qualityHandle_;
  std::string qualityLabel_;
};

#endif
