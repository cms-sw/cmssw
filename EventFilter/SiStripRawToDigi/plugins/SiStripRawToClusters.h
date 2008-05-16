#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClusters_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClusters_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/Common/interface/RefGetter.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToClustersLazyUnpacker.h"
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

  typedef edm::LazyGetter<SiStripCluster> LazyGetter;
  typedef edm::RefGetter<SiStripCluster> RefGetter;
  typedef SiStripRawToClustersLazyUnpacker LazyUnpacker;
  typedef SiStripRegionCabling::SubDet SubDet;

  SiStripRawToClusters( const edm::ParameterSet& );
  ~SiStripRawToClusters();
  
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 

  /** Raw data labels */
  std::string productLabel_;
  std::string productInstance_;

  /** Cabling */
  edm::ESHandle<SiStripRegionCabling> cabling_;

  /** Clusterizer Factory */
  SiStripClusterizerFactory* clusterizer_;
};

#endif //  EventFilter_SiStripRawToDigi_SiStripRawToClusters_H

