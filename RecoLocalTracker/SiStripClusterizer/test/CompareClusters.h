#ifndef CompareClusters_h
#define CompareClusters_h
class SiStripGain;
class SiStripNoises;
class SiStripQuality;
class SiStripDigi;
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <sstream>

class CompareClusters : public edm::EDAnalyzer {
  
  typedef edmNew::DetSetVector<SiStripCluster> input_t;
  
 public:
  
  CompareClusters(const edm::ParameterSet& conf);

 private:
  
  void analyze(const edm::Event&, const edm::EventSetup&);

  void show( uint32_t);
  std::string printDigis(uint32_t);
  static std::string printCluster(const SiStripCluster&);
  static bool identicalDSV(const input_t&,const input_t&);
  static bool identicalDetSet(const edmNew::DetSet<SiStripCluster>&,const edmNew::DetSet<SiStripCluster>&);
  static bool identicalClusters(const SiStripCluster&,const SiStripCluster&);

  std::stringstream message;
  edm::InputTag clusters1, clusters2, digis;
  edm::Handle<input_t> clusterHandle1, clusterHandle2;

  edm::Handle<edm::DetSetVector<SiStripDigi> > digiHandle;
  edm::ESHandle<SiStripGain> gainHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripQuality> qualityHandle;
};

#endif
