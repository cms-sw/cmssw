#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClustersDummyUnpacker_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClustersDummyUnpacker_H

//FWCore
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Data Formats
#include "DataFormats/SiStripCommon/test/stubs/SiStripLazyGetter.h"
#include "DataFormats/SiStripCommon/test/stubs/SiStripRefGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

//stl
#include <string>
#include <memory>
#include "boost/bind.hpp"

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripRawToClustersDummyUnpacker.h
   @class SiStripRawToClustersDummyUnpacker
*/

class SiStripRawToClustersDummyUnpacker : public edm::EDAnalyzer {
  
 public:

  typedef edm::DetSet<SiStripCluster> DetSet;
  typedef edm::SiStripRefGetter< SiStripCluster, edm::SiStripLazyGetter<SiStripCluster> > RefGetter;

  SiStripRawToClustersDummyUnpacker( const edm::ParameterSet& );
  ~SiStripRawToClustersDummyUnpacker();
  
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  
 private: 

  /** Input label */
  std::string inputModuleLabel_;
};

#endif //  EventFilter_SiStripRawToDigi_SiStripRawToClustersDummyRoI_H

