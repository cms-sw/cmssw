#ifndef ANALYSISALGOS_SiStripClusterInfoProducer_h
#define ANALYSISALGOS_SiStripClusterInfoProducer_h

#include <iostream> 
#include <memory>
#include <string>

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
//ES Data
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"

#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripCommonModeNoiseSubtractor.h"

class Event;
class EventSetup;
class SiStripRawDigi;
class SiStripDigi;
class SiStripCluster;
//class SiStripClusterInfo;

class SiStripClusterInfoProducer : public edm::EDProducer{
 public:

  explicit SiStripClusterInfoProducer(const edm::ParameterSet& conf);

  virtual ~SiStripClusterInfoProducer();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

  void cluster_algorithm(const edm::DetSetVector<SiStripCluster>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output);
  void digi_algorithm(const edm::DetSetVector<SiStripDigi>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output);
  void rawdigi_algorithm(const edm::DetSetVector<SiStripRawDigi>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output,std::string rawdigiLabel);
  void findNeigh(char* mode,std::vector< edm::DetSet<SiStripClusterInfo> >::iterator output_iter,std::vector<int16_t>& vadc,std::vector<int16_t>& vstrip);
  
 private:
  edm::ParameterSet conf_;
  uint16_t _NEIGH_STRIP_;

  SiStripCommonModeNoiseSubtractor* SiStripCommonModeNoiseSubtractor_;
  std::string CMNSubtractionMode_;
  bool validCMNSubtraction_;  
  SiStripPedestalsSubtractor* SiStripPedestalsSubtractor_;

  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripPedestals> pedestalsHandle;
  edm::ESHandle<SiStripGain> gainHandle;
};
#endif
