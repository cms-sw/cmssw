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
 * $Date: 2008/12/11 18:05:57 $
 * $Revision: 1.1 $
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

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <sys/time.h>

class EcalElectronicsMapper;
class EcalElectronicsMapping;
class DCCDataUnpacker;

class EcalRawToDigi : public edm::EDProducer{

 public:
  /**
   * Class constructor
   */
  explicit EcalRawToDigi(const edm::ParameterSet& ps);
  
  /**
   * Functions that are called by framework at each event
   */
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  
  /**
   * Class destructor
   */
  virtual ~EcalRawToDigi();
  
 private:

  //list of FEDs to unpack
  std::vector<int> fedUnpackList_;

  std::vector<int> orderedFedUnpackList_;
  std::vector<int> orderedDCCIdList_;
  
  uint numbXtalTSamples_;
  uint numbTriggerTSamples_;
  
  bool headerUnpacking_;
  bool srpUnpacking_;
  bool tccUnpacking_;
  bool feUnpacking_;
  bool memUnpacking_;
  bool syncCheck_;
  bool feIdCheck_;
  bool first_;
  bool put_;

  std::string dataLabel_ ; 

  // -- For regional unacking :
  bool REGIONAL_ ;
  edm::InputTag fedsLabel_ ;

  //an electronics mapper class 
  EcalElectronicsMapper * myMap_;

 
  //Ecal unpacker
  DCCDataUnpacker * theUnpacker_;

   
  
  uint nevts_; // NA: for testing
  double  RUNNING_TIME_, SETUP_TIME_;
  
  
};



#endif
