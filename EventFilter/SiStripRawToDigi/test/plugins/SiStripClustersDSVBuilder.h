#ifndef EventFilter_SiStripRawToDigi_SiStripClustersDSVBuilder_H
#define EventFilter_SiStripRawToDigi_SiStripClustersDSVBuilder_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/RefGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include <memory>
#include "boost/bind.hpp"

class SiStripClustersDSVBuilder : public edm::EDProducer {
  
 public:

  typedef edm::DetSet<SiStripCluster> DetSet;
  typedef edm::DetSetVector<SiStripCluster> DSV;
  typedef edm::LazyGetter<SiStripCluster> LazyGetter;
  typedef edm::RefGetter<SiStripCluster> RefGetter;

  SiStripClustersDSVBuilder( const edm::ParameterSet& );
  ~SiStripClustersDSVBuilder();
  
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 

  /** Input labels */
  edm::InputTag siStripLazyGetter_;
  edm::InputTag siStripRefGetter_;
};

#endif //  EventFilter_SiStripRawToDigi_SiStripClustersDSVBuilder_H
