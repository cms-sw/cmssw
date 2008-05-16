#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H

//FWCore
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Data Formats
#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

//CalibFormats
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"

//CommonTools
#include "CommonTools/SiStripClusterization/interface/SiStripClusterizerFactory.h"

//EventFilter
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h"

//Fed9U
#include "Fed9UUtils.hh"

//stl
#include <vector>

//#define USE_FED9U_EVENT_STREAMLINE

class SiStripRawToClustersLazyUnpacker : public edm::LazyUnpacker<SiStripCluster> {

 public:

  typedef edm::DetSet<SiStripCluster> DetSet;

  SiStripRawToClustersLazyUnpacker(const SiStripRegionCabling&,
				   const SiStripClusterizerFactory&,
				   const FEDRawDataCollection&); 
  
  virtual ~SiStripRawToClustersLazyUnpacker();

  virtual void fill(const uint32_t&, record_type&); 

 private:

  SiStripRawToClustersLazyUnpacker();

  //Raw data
  const FEDRawDataCollection* raw_;

  //Cabling
  const SiStripRegionCabling::Cabling* regions_;

  //Clusterizer Factory
  const SiStripClusterizerFactory* clusterizer_;

  //Fed9UEvent cache
#ifdef USE_FED9U_EVENT_STREAMLINE
  std::vector< Fed9U::Fed9UEventStreamLine* > fedEvents_;
#else
  std::vector< Fed9U::Fed9UEvent* > fedEvents_;
#endif

  //RawToDigi
  SiStripRawToDigiUnpacker rawToDigi_;

};

#endif //  EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H
