// Last commit: $Id: $
// Latest tag:  $Name: $
// Location:    $Source: $

#ifndef EventFilter_SiStripRawToDigi_interface_SiStripRawToClustersModule_H
#define EventFilter_SiStripRawToDigi_interface_SiStripRawToClustersModule_H

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Fed9UUtils.hh"
#include "boost/cstdint.hpp"
#include <string>
#include <vector>

class SiStripRawToDigiUnpacker;
class SiStripClusterizerFactory;
class SiStripDetCabling;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripRawToClusterModule.h
   @class SiStripRawToClusterModule 
   @author M.Wingham
   
   @brief A plug-in module that takes a FEDRawDataCollection as input
   from the Event and creates EDProducts containing StripClusters.
*/
class SiStripRawToClustersModule : public edm::EDProducer {
  
 public:
  
  SiStripRawToClustersModule( const edm::ParameterSet& );
  ~SiStripRawToClustersModule();
  
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob();
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 
  
  //zs-clusterizer algos...
  SiStripRawToDigiUnpacker* rawToDigi_;
  SiStripClusterizerFactory* clusterizer_;

  //Cabling
  edm::ESHandle<SiStripFedCabling> fedCabling_;
  SiStripDetCabling* detCabling_;

  //Input labels
  std::string productLabel_;
  std::string productInstance_;

  //Fed Unpacking
  int16_t headerBytes_;
  int16_t dumpFrequency_;
  int16_t triggerFedId_;
  bool useFedKey_;

  //Clusters container
  std::vector< edm::DetSet<SiStripCluster> > clusters_;

  //Fed9UEvent cache
  std::vector< Fed9U::Fed9UEvent* > fedEvents_;

};

#endif //  EventFilter_SiStripRawToDigi_interface_SiStripRawToClustersModule_H

