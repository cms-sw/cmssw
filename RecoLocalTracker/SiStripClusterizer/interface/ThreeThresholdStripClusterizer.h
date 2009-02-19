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
  ThreeThresholdStripClusterizer(float strip_thr, float seed_thr,float clust_thr, int max_holes, int max_bad=0, int max_adj=1);
  ~ThreeThresholdStripClusterizer();
  void init(const edm::EventSetup& es, std::string qualityLabel="", std::string thresholdLabel=""); 
  void clusterizeDetUnit(const edm::DetSet<SiStripDigi> &,    edmNew::DetSetVector<SiStripCluster>::FastFiller & output);
  void clusterizeDetUnit(const edmNew::DetSet<SiStripDigi> &, edmNew::DetSetVector<SiStripCluster>::FastFiller & output);
  
  struct InvalidChargeException : public cms::Exception { public: InvalidChargeException(const SiStripDigi&); };
 private:
  struct isSeed;
  struct thresholdGroup;
  struct DigiInfo;

  template<class T> void clusterizeDetUnitTemplate(const T & digis, edmNew::DetSetVector<SiStripCluster>::FastFiller& output);

  template<class T> T findClusterEdge(T,T) const;
  template<class T> bool clusterEdgeCondition(T,T,T) const;
  template<class T> bool aboveClusterThreshold(T,T) const;
  template<class T> void clusterize(T,T, edmNew::DetSetVector<SiStripCluster>::FastFiller& output);

  thresholdGroup* thresholds;
  DigiInfo* digiInfo;
  std::vector<uint16_t> amplitudes;
};



struct ThreeThresholdStripClusterizer::isSeed {
  bool operator()(const SiStripDigi& digi);
  isSeed(DigiInfo* i) : digiInfo(i) {}
  private: DigiInfo* digiInfo;
};
  

//This structure is planned to move into the event setup
struct ThreeThresholdStripClusterizer::thresholdGroup {
  thresholdGroup(float strip_thr, float seed_thr,float clust_thr, int max_holes, int max_bad=0, int max_adj=1)
    : strip(strip_thr), seed(seed_thr), cluster(clust_thr), holes(max_holes), bad(max_bad),adj(max_adj) {}
  uint8_t getMaxSequentialHoles(uint32_t) {return holes;}
  uint8_t getMaxSequentialBad(uint32_t) {return bad;}
  uint8_t getMaxAdjacentBad(uint32_t) {return adj;}
  float getClusterThreshold(uint32_t) {return cluster;}
  float getSeedThreshold(uint32_t, uint16_t) {return seed;}
  float getChannelThreshold(uint32_t,uint16_t) {return strip;}
  private:
  float strip, seed, cluster;
  uint8_t holes, bad, adj;
};

#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

struct ThreeThresholdStripClusterizer::DigiInfo {
  DigiInfo(const edm::EventSetup&, thresholdGroup* thresholds, std::string qualityLabel);
  bool isModuleUsable(uint32_t id) const {return qualityHandle->IsModuleUsable(id);}
  void setFastAccessDetId(uint32_t);
  uint32_t detId() const {return currentDetId;}
  uint8_t maxSequentialHoles() const {return thresholdHandle->getMaxSequentialHoles(currentDetId);}
  uint8_t maxSequentialBad()   const {return thresholdHandle->getMaxSequentialBad(currentDetId);}
  uint8_t maxAdjacentBad()     const {return thresholdHandle->getMaxAdjacentBad(currentDetId);}
  float clusterThreshold()     const {return thresholdHandle->getClusterThreshold(currentDetId);}
  float channelThreshold(const SiStripDigi& digi)const {return thresholdHandle->getChannelThreshold(currentDetId,digi.strip());}
  float seedThreshold(const SiStripDigi& digi)   const {return thresholdHandle->getSeedThreshold(currentDetId,digi.strip());}
  float noise(const SiStripDigi& digi)           const {return noiseHandle->getNoise(digi.strip(),noiseRange);}
  float gain(const SiStripDigi& digi)            const {return gainHandle->getStripGain(digi.strip(),gainRange);}
  bool isGoodStrip(const SiStripDigi& digi)      const {return !qualityHandle->IsStripBad(qualityRange, digi.strip());}
  bool isAboveSeed(const SiStripDigi& digi)      const {return digi.adc() >= static_cast<uint16_t>( noise(digi)*seedThreshold(digi) );}
  bool isAboveChannel(const SiStripDigi& digi)   const {return digi.adc() >= static_cast<uint16_t>( noise(digi)*channelThreshold(digi) );}
  bool includeInCluster(const SiStripDigi& digi) const {return isAboveChannel(digi) && isGoodStrip(digi);}
  bool anyGoodBetween(uint16_t,uint16_t) const;
  uint8_t nBadBeforeUpToMaxAdjacent(const SiStripDigi&) const;
  uint8_t nBadAfterUpToMaxAdjacent(const SiStripDigi&) const;
  uint16_t correctedCharge(const SiStripDigi&) const;
  
  private:
  edm::ESHandle<SiStripGain> gainHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripQuality> qualityHandle;
  thresholdGroup* thresholdHandle;
  
  uint32_t currentDetId;
  SiStripApvGain::Range gainRange;
  SiStripNoises::Range  noiseRange;
  SiStripQuality::Range qualityRange;
    
};

#endif
