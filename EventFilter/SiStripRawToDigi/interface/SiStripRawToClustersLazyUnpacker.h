#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerFactory.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include <vector>
#include <list>

/// tkonlinesw headers
#include "Fed9UUtils.hh"

class OldSiStripRawToClustersLazyUnpacker : public edm::LazyUnpacker<SiStripCluster> {

 public:

  typedef edm::DetSet<SiStripCluster> DetSet;

  OldSiStripRawToClustersLazyUnpacker(const SiStripRegionCabling&, const SiStripClusterizerFactory&, const FEDRawDataCollection&); 
  
  virtual ~OldSiStripRawToClustersLazyUnpacker();

  virtual void fill(const uint32_t&, record_type&); 

 private:

  OldSiStripRawToClustersLazyUnpacker();

  /// raw data
  const FEDRawDataCollection* raw_;

  /// cabling
  const SiStripRegionCabling::Cabling* regions_;

  /// clusterizer factory
  const SiStripClusterizerFactory* clusterizer_;

  /// Fed9UEvent cache
  std::vector< Fed9U::Fed9UEvent* > fedEvents_;

  /// raw-to-digi
  OldSiStripRawToDigiUnpacker rawToDigi_;

  /// Cache of buffers pointed to by FED9UEvent
  std::list<FEDRawData> fedRawData_;

};

namespace sistrip {
  
  class RawToClustersLazyUnpacker : public edm::LazyUnpacker<SiStripCluster> {
    
  public:
    
    typedef edm::DetSet<SiStripCluster> DetSet;
    
    RawToClustersLazyUnpacker(const SiStripRegionCabling&, StripClusterizerAlgorithm&, SiStripRawProcessingAlgorithms&, const FEDRawDataCollection&, bool = false); 
    
    virtual ~RawToClustersLazyUnpacker();
    
    virtual void fill(const uint32_t&, record_type&); 
    
  private:
    
    /// private default constructor
    RawToClustersLazyUnpacker();
    
    /// raw data
    const FEDRawDataCollection* raw_;
    
    /// cabling
    const SiStripRegionCabling::Cabling* regions_;
    
    /// clusterizer algorithm
    StripClusterizerAlgorithm* const clusterizer_;
    
    /// raw processing algorithms
    SiStripRawProcessingAlgorithms* const rawAlgos_;
    
    /// FED event cache
    std::vector< sistrip::FEDBuffer* > buffers_;
    
    /// raw-to-digi
    RawToDigiUnpacker rawToDigi_;
    
    /// dump frequency
    bool dump_;
    
    /// FED mode
    sistrip::FEDReadoutMode mode_;
    
  };
  
}

#endif ///  EventFilter_SiStripRawToDigi_SiStripRawToClustersLazyUnpacker_H
