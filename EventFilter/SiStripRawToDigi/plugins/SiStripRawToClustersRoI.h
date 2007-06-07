#ifndef EventFilter_SiStripRawToDigi_SiStripRawToClustersRoI_H
#define EventFilter_SiStripRawToDigi_SiStripRawToClustersRoI_H

//FWCore
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Data Formats
#include "DataFormats/SiStripCommon/interface/SiStripRefGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

//CalibFormats
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"

//stl
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

  typedef edm::SiStripRefGetter< SiStripCluster, edm::SiStripLazyGetter<SiStripCluster> > RefGetter;

  SiStripRawToClustersRoI( const edm::ParameterSet& );
  ~SiStripRawToClustersRoI();
  
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 

  /** Method defining regions of interest randomly */
  void random(SiStripRegionCabling::Regions&) const;

  /** Method defining ALL regions of interest */
  void all(SiStripRegionCabling::Regions&) const;
  
  /** Method defining regions of interest by superclusters */
  void superclusters(const reco::SuperClusterCollection&,
		     SiStripRegionCabling::Regions&) const;

  /** Input module label of SiStripLazyGetter */
  std::string inputModuleLabel_;

  /** Cabling */
  edm::ESHandle<SiStripRegionCabling> cabling_;

  /** Booleans to define objects of interest */
  bool random_;
  bool all_;
  bool electron_;

  /** dR to define regions of interest around physics objects */
  double dR_;

};

#endif //  EventFilter_SiStripRawToDigi_SiStripRawToClustersRoI_H

