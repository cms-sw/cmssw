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
  void init(const edm::EventSetup& es, std::string qualityLabel=""); 

  typedef edmNew::DetSetVector<SiStripCluster>::FastFiller output_t;
  void clusterizeDetUnit(const    edm::DetSet<SiStripDigi>&, output_t&);
  void clusterizeDetUnit(const edmNew::DetSet<SiStripDigi>&, output_t&);  

  struct InvalidChargeException : public cms::Exception { public: InvalidChargeException(const SiStripDigi&); };


 private:

  template<class T> void clusterizeDetUnit_(const T &, output_t&);

  //state of the candidate cluster
  std::vector<uint16_t> ADCs;  
  uint16_t lastStrip;
  float noiseSquared;
  bool candidateHasSeed;

  //constant methods with state information
  uint16_t firstStrip() const {return lastStrip - ADCs.size() + 1;}
  bool candidateEnded(uint16_t) const;
  bool candidateAccepted() const;

  //state modification methods
  void clearCandidate() { candidateHasSeed = false;  noiseSquared = 0;  ADCs.clear();}
  void addToCandidate(const SiStripDigi&);
  void appendBadNeighborsToCandidate();
  class applyGain; //functor

  class ESinfo; ESinfo* info;

};



class ThreeThresholdStripClusterizer::applyGain {
 public:
  uint16_t operator()(uint16_t);
  applyGain(const ESinfo* es, uint16_t firststrip) : info(es), strip(firststrip) {}
 private: 
  const ESinfo* info; 
  uint16_t strip;
};




#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

class ThreeThresholdStripClusterizer::ESinfo {

 public:

  ESinfo(float chan, float seed, float clust, uint8_t holes, uint8_t bad, uint8_t adj) : 
    ChannelThreshold(chan), SeedThreshold(seed), ClusterThresholdSquared(clust*clust),
    MaxSequentialHoles(holes), MaxSequentialBad(bad), MaxAdjacentBad(adj),
    noise_cache_id(0), gain_cache_id(0), quality_cache_id(0)
  {}
  const float ChannelThreshold, SeedThreshold, ClusterThresholdSquared;
  const uint8_t MaxSequentialHoles, MaxSequentialBad, MaxAdjacentBad;
  
  float noise(const uint16_t strip) const {return noiseHandle->getNoise(strip,noiseRange);}
  float gain(const uint16_t strip)  const {return gainHandle->getStripGain(strip,gainRange);}
  bool bad(const uint16_t strip)    const {return qualityHandle->IsStripBad(qualityRange, strip);}
  bool allBadBetween(uint16_t a,uint16_t b) const {while(++a<b && bad(a)); return a==b;}
  bool isModuleUsable(uint32_t id)  const {return qualityHandle->IsModuleUsable(id);}
  
  void setDetId(uint32_t);
  SiStripApvGain::Range gainRange;
  SiStripNoises::Range  noiseRange;
  SiStripQuality::Range qualityRange;
  edm::ESHandle<SiStripGain> gainHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripQuality> qualityHandle;
  uint32_t noise_cache_id, gain_cache_id, quality_cache_id;

};

#endif
