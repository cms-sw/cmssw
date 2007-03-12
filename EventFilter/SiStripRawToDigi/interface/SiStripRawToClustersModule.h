#ifndef EventFilter_SiStripRawToDigi_interface_SiStripRawToClustersModule_H
#define EventFilter_SiStripRawToDigi_interface_SiStripRawToClustersModule_H

#include "FWCore/Framework/interface/EDProducer.h"
#include <string>

#include "CommonTools/SiStripClusterization/interface/SiStripClusterizerFactory.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigiCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiUnpacker.h"
//Fed9U
#include "Fed9UUtils.hh"

#include <memory>
#include <sstream>
#include <iomanip>
#include <string>

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripRawToClusterModule.h
   @class SiStripRawToClusterModule 
   
   @brief A plug-in module that takes a FEDRawDataCollection as input
   from the Event and creates EDProducts containing StripClusters.
*/
class SiStripRawToClustersModule : public edm::EDProducer {
  
  typedef std::vector<edm::ParameterSet> Parameters;
  
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

  //Fed Unpacking
  int16_t headerBytes_;
  int16_t dumpFrequency_;
  int16_t triggerFedId_;
  bool useFedKey_;

  //Fed9UEvent cache
  std::vector< Fed9U::Fed9UEvent* > fedEvents_;

};

#endif //  EventFilter_SiStripRawToDigi_interface_SiStripRawToClustersModule_H

