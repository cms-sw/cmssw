#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H

//FWCore
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Data Formats
#include "DataFormats/SiStripCommon/interface/SiStripLazyGetter.h"
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

class SiStripRawToClustersLazyUnpacker : public edm::SiStripLazyUnpacker<SiStripCluster> {

 public:

  typedef edm::DetSet<SiStripCluster> DetSet;
  typedef edm::SiStripLazyUnpacker<SiStripCluster> Base;

  SiStripRawToClustersLazyUnpacker(const SiStripRegionCabling&,
				   const SiStripClusterizerFactory&,
				   const FEDRawDataCollection&); 
  
  ~SiStripRawToClustersLazyUnpacker();

  virtual void fill(uint32_t&); 

 private:

  SiStripRawToClustersLazyUnpacker();

  //Raw data
  const FEDRawDataCollection* raw_;

  //Cabling
  const SiStripRegionCabling::Cabling* regions_;

  //Clusterizer Factory
  const SiStripClusterizerFactory* clusterizer_;

  //Fed9UEvent cache
  std::vector< Fed9U::Fed9UEvent* > fedEvents_;

  //RawToDigi
  SiStripRawToDigiUnpacker rawToDigi_;

};

#endif //  EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H
