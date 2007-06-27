#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClusters_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClusters_H

#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToClustersLazyUnpacker.h"

//FWCore
#include "FWCore/Framework/interface/EDProducer.h"

//Data Formats
#include "DataFormats/SiStripCommon/interface/SiStripLazyGetter.h"
#include "DataFormats/SiStripCommon/interface/SiStripRefGetter.h"

//CalibFormats
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"

//CalibTracker
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"

//stl
#include <string>
#include <memory>
#include "boost/bind.hpp"

class SiStripClusterizerFactory;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripRawToClusters.h
   @class SiStripRawToClusters
*/

class SiStripRawToClusters : public edm::EDProducer {
  
 public:

  typedef edm::SiStripLazyGetter<SiStripCluster> LazyGetter;
  typedef edm::SiStripRefGetter<SiStripCluster> RefGetter;
  typedef SiStripRawToClustersLazyUnpacker LazyUnpacker;

  SiStripRawToClusters( const edm::ParameterSet& );
  ~SiStripRawToClusters();
  
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 

  //Record of all region numbers
  std::vector<uint32_t> allregions_;

  //Raw data labels
  std::string productLabel_;
  std::string productInstance_;

  //Cabling
  edm::ESHandle<SiStripRegionCabling> cabling_;

  //Clusterizer Factory
  SiStripClusterizerFactory* clusterizer_;

  //Fed
  int16_t dumpFrequency_;
  int16_t triggerFedId_;
};

#endif //  EventFilter_SiStripRawToDigi_SiStripRawToClusters_H

