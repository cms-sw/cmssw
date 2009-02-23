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

  typedef edmNew::DetSetVector<SiStripCluster>::FastFiller output_t;
  void clusterizeDetUnit(const    edm::DetSet<SiStripDigi>&, output_t&);
  void clusterizeDetUnit(const edmNew::DetSet<SiStripDigi>&, output_t&);  

  struct InvalidChargeException : public cms::Exception { public: InvalidChargeException(const SiStripDigi&); };
 private:
  template<class T> void clusterizeDetUnit_(const T &, output_t&);
  struct ESinfo; 
  struct applyGain;

  bool found() const;
  bool edgeCondition(uint16_t) const;

  void clear() { foundSeed = false;  noise2 = 0;  amp.clear();}
  void record(const SiStripDigi&);
  void appendBadNeighbors();  

  uint16_t first() const {return last - amp.size() + 1;}
  uint16_t last;
  float noise2;
  std::vector<uint16_t> amp;  
  bool foundSeed;
  ESinfo* info;
};


struct ThreeThresholdStripClusterizer::applyGain {
  uint16_t operator()(uint16_t);
  applyGain(const ESinfo* es, uint16_t firststrip) : info(es), strip(firststrip) {}
  private: const ESinfo* info; uint16_t strip;
};



#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

struct ThreeThresholdStripClusterizer::ESinfo {
  ESinfo(float chan, float seed, float clust, uint8_t holes, uint8_t bad, uint8_t adj) :
    Channel(chan), Seed(seed), Cluster2(clust*clust),
       MaxSequentialHoles(holes), MaxSequentialBad(bad), MaxAdjacentBad(adj)
  {}
  
  void setDetId(uint32_t);
  bool isModuleUsable(uint32_t id)  const {return qualityHandle->IsModuleUsable(id);}
  float noise(const uint16_t strip) const {return noiseHandle->getNoise(strip,noiseRange);}
  float gain(const uint16_t strip)  const {return gainHandle->getStripGain(strip,gainRange);}
  bool bad(const uint16_t strip)    const {return qualityHandle->IsStripBad(qualityRange, strip);}
  bool anyGoodBetween(uint16_t a,uint16_t b) const {while(a<b && bad(a)) a++; return a!=b;}

  edm::ESHandle<SiStripGain> gainHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripQuality> qualityHandle;
  
  SiStripApvGain::Range gainRange;
  SiStripNoises::Range  noiseRange;
  SiStripQuality::Range qualityRange;

  const float Channel, Seed, Cluster2;
  const uint8_t MaxSequentialHoles, MaxSequentialBad, MaxAdjacentBad;
};

#endif
