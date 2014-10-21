#ifndef _ECALRAWTODIGIDEV_H_ 
#define _ECALRAWTODIGIDEV_H_ 

/*
 *\ Class EcalRawToDigi
 *
 * This class takes unpacks ECAL raw data 
 * produces digis and raw data format prolblems reports
 *
 * \file EcalRawToDigi.h
 *
 * \author N. Almeida
 * \author G. Franzoni
 *
*/

#include <iostream>                                 

#include "EventFilter/EcalRawToDigi/interface/DCCRawDataDefinitions.h"

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Framework/interface/ESWatcher.h>
#include "DataFormats/EcalRawData/interface/EcalListOfFEDS.h"
#include <sys/time.h>

class EcalElectronicsMapper;
class EcalElectronicsMapping;
class DCCDataUnpacker;

class EcalRawToDigi : public edm::stream::EDProducer<>{

 public:
  /**
   * Class constructor
   */
  explicit EcalRawToDigi(const edm::ParameterSet& ps);
  
  /**
   * Functions that are called by framework at each event
   */
  virtual void produce(edm::Event& e, const edm::EventSetup& c) override;
  
  // function called at start of each run
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  
  /**
   * Class destructor
   */
  virtual ~EcalRawToDigi();

 
  edm::ESWatcher<EcalMappingRcd> watcher_;

  
 private:

  //list of FEDs to unpack
  std::vector<int> fedUnpackList_;

  std::vector<int> orderedFedUnpackList_;
  std::vector<int> orderedDCCIdList_;
  
  unsigned int numbXtalTSamples_;
  unsigned int numbTriggerTSamples_;
  
  bool headerUnpacking_;
  bool srpUnpacking_;
  bool tccUnpacking_;
  bool feUnpacking_;
  bool memUnpacking_;
  bool syncCheck_;
  bool feIdCheck_;
  bool forceToKeepFRdata_;
  bool first_;
  bool put_;

  
  edm::EDGetTokenT<FEDRawDataCollection> dataToken_;
  edm::EDGetTokenT<EcalListOfFEDS> fedsToken_;  

  // -- For regional unacking :
  bool REGIONAL_ ;
    

  //an electronics mapper class 
  EcalElectronicsMapper * myMap_;
  
  //Ecal unpacker
  DCCDataUnpacker * theUnpacker_;
  
  unsigned int nevts_; // NA: for testing
  double  RUNNING_TIME_, SETUP_TIME_;
};

#endif
