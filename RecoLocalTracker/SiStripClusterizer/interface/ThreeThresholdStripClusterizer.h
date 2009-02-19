#ifndef RECOLOCALTRACKER_SISTRIPCLUSTERIZER_THREETHRESHOLDSTRIPCLUSTERIZER_H
#define RECOLOCALTRACKER_SISTRIPCLUSTERIZER_THREETHRESHOLDSTRIPCLUSTERIZER_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include <string>
#include <vector>

class ThreeThresholdStripClusterizer {
 public:
  ThreeThresholdStripClusterizer(float channel, float seed,float cluster, int holes, int bad=0, int adj=1);
  ~ThreeThresholdStripClusterizer();
  void init(const edm::EventSetup& es, std::string qualityLabel="", std::string thresholdLabel=""); 

  void clusterizeDetUnit(const    edm::DetSet<SiStripDigi> &digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output);
  void clusterizeDetUnit(const edmNew::DetSet<SiStripDigi> &digis, edmNew::DetSetVector<SiStripCluster>::FastFiller & output);  
  struct InvalidChargeException : public cms::Exception { public: InvalidChargeException(const SiStripDigi&); };

 private:
  struct ExtendedDigi; 
  typedef std::vector<ExtendedDigi>::const_iterator           iter_t;
  typedef std::vector<ExtendedDigi>::const_reverse_iterator  riter_t;

  template<class T> void clusterizeDetUnit_(const T &, edmNew::DetSetVector<SiStripCluster>::FastFiller&);
  template<class T> T findClusterEdge(T,T) const;
  template<class T> bool clusterEdgeCondition(T,T,T) const;
  bool aboveClusterThreshold(iter_t,iter_t) const;
  void clusterize(iter_t,iter_t, edmNew::DetSetVector<SiStripCluster>::FastFiller&);
 
  struct isSeed;
  struct ExtendedDigiFactory;
  struct Thresholds; Thresholds*  thresholds;
  struct ESinfo;         ESinfo*  esinfo;
  std::vector<uint16_t>           amplitudes;
  std::vector<ExtendedDigi>       extDigis;
};
  



struct ThreeThresholdStripClusterizer::Thresholds {
  Thresholds(float strip_thr, float seed_thr,float clust_thr, int max_holes, int max_bad=0, int max_adj=1)
    : Channel(strip_thr), Seed(seed_thr), Cluster(clust_thr), MaxSequentialHoles(max_holes), MaxSequentialBad(max_bad),MaxAdjacentBad(max_adj) {}
  const float Channel, Seed, Cluster;
  const uint8_t MaxSequentialHoles, MaxSequentialBad, MaxAdjacentBad;
};

#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

struct ThreeThresholdStripClusterizer::ESinfo {
  ESinfo(const edm::EventSetup&, std::string qualityLabel);
  void setDetId(uint32_t);
  uint32_t detId() const {return currentDetId;}

  bool isModuleUsable(uint32_t id)  const {return qualityHandle->IsModuleUsable(id);}
  float noise(const uint16_t strip) const {return noiseHandle->getNoise(strip,noiseRange);}
  float gain(const uint16_t strip)  const {return gainHandle->getStripGain(strip,gainRange);}
  bool IsBad(const uint16_t strip)  const {return qualityHandle->IsStripBad(qualityRange, strip);}
  bool anyGoodBetween(uint16_t,uint16_t) const;
  uint8_t badAdjacent(const uint16_t& strip,const uint8_t& max, int8_t direction) const;

  private:
  edm::ESHandle<SiStripGain> gainHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripQuality> qualityHandle;
  
  uint32_t currentDetId;
  SiStripApvGain::Range gainRange;
  SiStripNoises::Range  noiseRange;
  SiStripQuality::Range qualityRange;  
};




struct ThreeThresholdStripClusterizer::ExtendedDigi {
  ExtendedDigi(const SiStripDigi& digi_, float noise_, bool bad, float chan, float seed) 
    : digi(digi_), noise(noise_), aboveSeed( !bad && digi_.adc() >= static_cast<uint16_t>(noise_*seed)) {
    aboveChannel = ( aboveSeed ||	     !bad && digi_.adc() >= static_cast<uint16_t>(noise_*chan));
  }
  const SiStripDigi&   digi;
  const float          noise;
  const bool           aboveSeed;
  bool                 aboveChannel;

  uint16_t strip() const {return digi.strip();}
  uint16_t adc()   const {return digi.adc();}
  uint16_t correctedCharge(ESinfo*) const;
  const ExtendedDigi& operator=(const ExtendedDigi& d) {return d;}
};

struct ThreeThresholdStripClusterizer::ExtendedDigiFactory {
  const ExtendedDigi& operator()(const SiStripDigi& digi) { 
    return *(new ExtendedDigi( digi, e->noise(digi.strip()), e->IsBad(digi.strip()), t->Channel, t->Seed));
  }
  ExtendedDigiFactory(ESinfo* e, Thresholds* t) : e(e),t(t) {}
  private: ESinfo* e; Thresholds* t;
};

struct ThreeThresholdStripClusterizer::isSeed { 
  bool operator()(const ExtendedDigi& digi) { 
    return digi.aboveSeed; 
  }
  isSeed() {}
};

#endif
