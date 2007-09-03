#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClustersRoI_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClustersRoI_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiStripCommon/interface/SiStripRefGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include <string>
#include <memory>
#include "boost/bind.hpp"

/**
   @file EventFilter/SiStripRawToDigi/test/plugins/SiStripRawToClustersRoI.h
   @class SiStripRawToClustersRoI
   @author M.Wingham
*/

class SiStripRawToClustersRoI : public edm::EDProducer {
  
 public:

  typedef edm::SiStripLazyGetter<SiStripCluster> LazyGetter;
  typedef edm::SiStripRefGetter<SiStripCluster> RefGetter;

  SiStripRawToClustersRoI( const edm::ParameterSet& );
  ~SiStripRawToClustersRoI();
  
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 

  /** Method defining allowed physical layer numbers */
  bool physicalLayer(SiStripRegionCabling::SubDet&, uint32_t&) const;

  /** Method defining regions of interest randomly */
  void random(RefGetter&, edm::Handle<LazyGetter>&) const;

  /** Method defining ALL regions of interest */
  void all(RefGetter&, edm::Handle<LazyGetter>&) const;
  
  /** Method defining regions of interest by superclusters */
  void superclusters(const reco::SuperClusterCollection&, RefGetter&, edm::Handle<LazyGetter>&) const;

  /** Input module label of SiStripLazyGetter */
  std::string inputModuleLabel_;

  /** Cabling */
  edm::ESHandle<SiStripRegionCabling> cabling_;

  /** Layers of SST to unpack (from innermost) */
  int nlayers_;

  /** Booleans to define objects of interest */
  bool random_;
  bool all_;
  bool electron_;

  /** deta/dphi to define regions of interest around physics objects */
  double deta_;
  double dphi_;
};

#endif //  EventFilter_SiStripRawToDigi_SiStripRawToClustersRoI_H

