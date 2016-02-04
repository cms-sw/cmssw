#ifndef EventFilter_SiStripRawToDigi_SiStripTrivialClusterSource_H
#define EventFilter_SiStripRawToDigi_SiStripTrivialClusterSource_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "TRandom.h"
#include <vector>

 /**
    @file EventFilter/SiStripRawToDigi/test/plugins/SiStripTrivialClusterSource.h
    @class SiStripTrivialClusterSource
    @brief Creates a DetSetVector of SiStripDigis created using random
    number generators and attaches the collection to the Event. Allows
    to test the final DigiToRaw and RawToDigi/RawToCluster converters.  
 */

class SiStripTrivialClusterSource : public edm::EDProducer {
  
 public:
  
  SiStripTrivialClusterSource(const edm::ParameterSet&);
  ~SiStripTrivialClusterSource();
  
  virtual void beginRun( edm::Run&, const edm::EventSetup& );
  virtual void endJob();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private: 

  /** Check for space in module */
  bool available(const edm::DetSet<SiStripDigi>&, const uint16_t, const uint32_t);

  /** Add cluster to module */
  void addcluster(edm::DetSet<SiStripDigi>&, const uint16_t, const uint16_t);

  /** Configurables */
  double minocc_;
  double maxocc_;
  double mincluster_;
  double maxcluster_;
  uint16_t separation_;

  /** Setup */
  edm::ESHandle<SiStripDetCabling> cabling_;
  std::vector<uint32_t> detids_;
  uint32_t nstrips_;

  /** Random */
  TRandom random_; 
};

#endif // EventFilter_SiStripRawToDigi_SiStripTrivialClusterSource_H

