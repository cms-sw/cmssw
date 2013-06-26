#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClusters_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClusters_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/Common/interface/RefGetter.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToClustersLazyUnpacker.h"
#include <string>
#include <memory>
#include "boost/bind.hpp"

class SiStripClusterizerFactory;
class SiStripRegionCabling;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripRawToClusters.h
   @class sistrip::RawToClusters
*/

namespace sistrip {

  class RawToClusters : public edm::EDProducer {
    
  public:
    
    typedef edm::LazyGetter<SiStripCluster> LazyGetter;
    typedef edm::RefGetter<SiStripCluster> RefGetter;
    typedef RawToClustersLazyUnpacker LazyUnpacker;
    typedef SiStripRegionCabling::SubDet SubDet;
    
    RawToClusters( const edm::ParameterSet& );
    ~RawToClusters();
    
    virtual void beginRun( const edm::Run&, const edm::EventSetup& ) override;
    virtual void produce( edm::Event&, const edm::EventSetup& ) override;
    
  private: 
    
    void updateCabling( const edm::EventSetup& setup );
    
    edm::InputTag productLabel_;
    const SiStripRegionCabling* cabling_;
    uint32_t cacheId_;
    std::auto_ptr<StripClusterizerAlgorithm> clusterizer_;
    std::auto_ptr<SiStripRawProcessingAlgorithms> rawAlgos_;

    // March 2012: add flag for disabling APVe check in configuration
    bool doAPVEmulatorCheck_; 

  };
  
}

#endif // EventFilter_SiStripRawToDigi_SiStripRawToClusters_H
