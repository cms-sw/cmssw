#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "CommonTools/SiStripClusterization/interface/SiStripClusterizerFactory.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h"
#include "Fed9UUtils.hh"
#include <vector>

class SiStripRawToClustersLazyUnpacker : public edm::LazyUnpacker<SiStripCluster> {

 public:

  typedef edm::DetSet<SiStripCluster> DetSet;

  SiStripRawToClustersLazyUnpacker(const SiStripRegionCabling&, const SiStripClusterizerFactory&, const FEDRawDataCollection&); 
  
  virtual ~SiStripRawToClustersLazyUnpacker();

  virtual void fill(const uint32_t&, record_type&); 

 private:

  SiStripRawToClustersLazyUnpacker();

  /// Raw data
  const FEDRawDataCollection* raw_;

  /// Cabling
  const SiStripRegionCabling::Cabling* regions_;

  /// Clusterizer Factory
  const SiStripClusterizerFactory* clusterizer_;

  /// Fed9UEvent cache
  std::vector< Fed9U::Fed9UEvent* > fedEvents_;

  /// Fed9UEvent readout mode
  std::vector<sistrip::FedReadoutMode> fedModes_;

  /// RawToDigi
  SiStripRawToDigiUnpacker rawToDigi_;
};

#endif //  EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H
