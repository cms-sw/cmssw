/*
 * \file EBRawDataTask.cc
 *
 * $Date: 2008/07/28 19:06:44 $
 * $Revision: 1.4 $
 * \author E. Di Marco
 *
*/

#include <iostream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/FEDRawData/src/fed_header.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include "DQM/EcalBarrelMonitorTasks/interface/EBRawDataTask.h"

using namespace cms;
using namespace edm;
using namespace std;

EBRawDataTask::EBRawDataTask(const ParameterSet& ps) {

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");
  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  GTEvmSource_ =  ps.getParameter<edm::InputTag>("GTEvmSource");

  meEBCRCErrors_ = 0;
  meEBRunNumberErrors_ = 0;
  meEBL1AErrors_ = 0;
  meEBOrbitNumberErrors_ = 0;
  meEBBunchCrossingErrors_ = 0;
  meEBTriggerTypeErrors_ = 0;

}

EBRawDataTask::~EBRawDataTask() {
}

void EBRawDataTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBRawDataTask");
    dqmStore_->rmdir(prefixME_ + "/EBRawDataTask");
  }

  Numbers::initGeometry(c, false);

}

void EBRawDataTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EBRawDataTask::endRun(const Run& r, const EventSetup& c) {

}

void EBRawDataTask::reset(void) {

  if ( meEBCRCErrors_ ) meEBCRCErrors_->Reset();
  if ( meEBRunNumberErrors_ ) meEBRunNumberErrors_->Reset();
  if ( meEBL1AErrors_ ) meEBL1AErrors_->Reset();
  if ( meEBOrbitNumberErrors_ ) meEBOrbitNumberErrors_->Reset();
  if ( meEBBunchCrossingErrors_ ) meEBBunchCrossingErrors_->Reset();
  if ( meEBTriggerTypeErrors_ ) meEBTriggerTypeErrors_->Reset();

}

void EBRawDataTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBRawDataTask");

    sprintf(histo, "EBRDT CRC errors");
    meEBCRCErrors_ = dqmStore_->book1D(histo, histo, 36, 1, 37);
    for (int i = 0; i < 36; i++) {
      meEBCRCErrors_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    sprintf(histo, "EBRDT run number errors");
    meEBRunNumberErrors_ = dqmStore_->book1D(histo, histo, 36, 1, 37); 
    for (int i = 0; i < 36; i++) {
      meEBRunNumberErrors_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    sprintf(histo, "EBRDT L1A errors");
    meEBL1AErrors_ = dqmStore_->book1D(histo, histo, 36, 1, 37); 
    for (int i = 0; i < 36; i++) {
      meEBL1AErrors_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    sprintf(histo, "EBRDT orbit number errors");
    meEBOrbitNumberErrors_ = dqmStore_->book1D(histo, histo, 36, 1, 37); 
    for (int i = 0; i < 36; i++) {
      meEBOrbitNumberErrors_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    sprintf(histo, "EBRDT bunch crossing errors");
    meEBBunchCrossingErrors_ = dqmStore_->book1D(histo, histo, 36, 1, 37); 
    for (int i = 0; i < 36; i++) {
      meEBBunchCrossingErrors_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

    sprintf(histo, "EBRDT trigger type errors");
    meEBTriggerTypeErrors_ = dqmStore_->book1D(histo, histo, 36, 1, 37); 
    for (int i = 0; i < 36; i++) {
      meEBTriggerTypeErrors_->setBinLabel(i+1, Numbers::sEB(i+1).c_str(), 1);
    }

  }

}

void EBRawDataTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBRawDataTask");

    if ( meEBCRCErrors_ ) dqmStore_->removeElement( meEBCRCErrors_->getName() );
    meEBCRCErrors_ = 0;

    if ( meEBRunNumberErrors_ ) dqmStore_->removeElement( meEBRunNumberErrors_->getName() );
    meEBRunNumberErrors_ = 0;

    if ( meEBL1AErrors_ ) dqmStore_->removeElement( meEBL1AErrors_->getName() );
    meEBL1AErrors_ = 0;

    if ( meEBOrbitNumberErrors_ ) dqmStore_->removeElement( meEBOrbitNumberErrors_->getName() );
    meEBOrbitNumberErrors_ = 0;

    if ( meEBBunchCrossingErrors_ ) dqmStore_->removeElement( meEBBunchCrossingErrors_->getName() );
    meEBBunchCrossingErrors_ = 0;

    if ( meEBTriggerTypeErrors_ ) dqmStore_->removeElement( meEBTriggerTypeErrors_->getName() );
    meEBTriggerTypeErrors_ = 0;

  }

  init_ = false;

}

void EBRawDataTask::endJob(void) {

  LogInfo("EBRawDataTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBRawDataTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  int GT_L1A=0, GT_OrbitNumber=0, GT_BunchCrossing=0, GT_TriggerType=0;

  edm::Handle<FEDRawDataCollection> allFedRawData;

  int gtFedDataSize = 0;
  bool GT_OrbitNumber_Present = false;

  int ECALDCC_L1A_MostFreqId = -1;
  int ECALDCC_OrbitNumber_MostFreqId = -1;
  int ECALDCC_BunchCrossing_MostFreqId = -1;
  int ECALDCC_TriggerType_MostFreqId = -1;
    
  if ( e.getByLabel(FEDRawDataCollection_, allFedRawData) ) {

    // GT FED data
    const FEDRawData& gtFedData = allFedRawData->FEDData(812);
    
    gtFedDataSize = gtFedData.size()/sizeof(uint64_t);
    
    if ( gtFedDataSize > 0 ) {
      
      FEDHeader header(gtFedData.data());
      
      GT_L1A = header.lvl1ID();
      GT_BunchCrossing = header.bxID();
      GT_TriggerType = header.triggerType();
      
    } 

    Handle<L1GlobalTriggerEvmReadoutRecord> GTEvmReadoutRecord;
    e.getByLabel(GTEvmSource_, GTEvmReadoutRecord);
    
    if (!GTEvmReadoutRecord.isValid()) {
      edm::LogWarning("EBRawDataTask") << "L1GlobalTriggerEvmReadoutRecord with label "
				       << GTEvmSource_.label() << " not available";
    } else {
      
      L1GtfeWord gtfeEvmWord = GTEvmReadoutRecord->gtfeWord();
      int gtfeEvmActiveBoards = gtfeEvmWord.activeBoards();
      
      if( gtfeEvmActiveBoards & (1<<TCS) ) { // if TCS present in the record
	
	GT_OrbitNumber_Present = true;
	
	L1TcsWord tcsWord = GTEvmReadoutRecord->tcsWord();
	
	GT_OrbitNumber = tcsWord.orbitNr();
	
      }
    }

    if ( gtFedDataSize == 0 || !GT_OrbitNumber_Present ) {
      // use the most frequent among the ECAL FEDs
      
      std::map<int,int> ECALDCC_L1A_FreqMap;
      std::map<int,int> ECALDCC_OrbitNumber_FreqMap;
      std::map<int,int> ECALDCC_BunchCrossing_FreqMap;
      std::map<int,int> ECALDCC_TriggerType_FreqMap;
      
      int ECALDCC_L1A_MostFreqCounts = 0;
      int ECALDCC_OrbitNumber_MostFreqCounts = 0;
      int ECALDCC_BunchCrossing_MostFreqCounts = 0;
      int ECALDCC_TriggerType_MostFreqCounts = 0;
      
      Handle<EcalRawDataCollection> dcchs;
      
      if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {
	
	if ( dcchs.isValid() ) {
	  
	  for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {
	    
	    EcalDCCHeaderBlock dcch = (*dcchItr);
	
	    if ( Numbers::subDet( dcch ) != EcalBarrel ) continue;
	    
	    int ECALDCC_L1A = dcch.getLV1();
	    int ECALDCC_OrbitNumber = dcch.getOrbit();
	    int ECALDCC_BunchCrossing = dcch.getBX();
	    int ECALDCC_TriggerType = dcch.getBasicTriggerType();

	    ++ECALDCC_L1A_FreqMap[ECALDCC_L1A];
	    ++ECALDCC_OrbitNumber_FreqMap[ECALDCC_OrbitNumber];
	    ++ECALDCC_BunchCrossing_FreqMap[ECALDCC_BunchCrossing];
	    ++ECALDCC_TriggerType_FreqMap[ECALDCC_TriggerType];
	    
	    if ( ECALDCC_L1A_FreqMap[ECALDCC_L1A] > ECALDCC_L1A_MostFreqCounts ) {
	      ECALDCC_L1A_MostFreqCounts = ECALDCC_L1A_FreqMap[ECALDCC_L1A];
	      ECALDCC_L1A_MostFreqId = ECALDCC_L1A;
	    }
	    
	    if ( ECALDCC_OrbitNumber_FreqMap[ECALDCC_OrbitNumber] > ECALDCC_OrbitNumber_MostFreqCounts ) {
	      ECALDCC_OrbitNumber_MostFreqCounts = ECALDCC_OrbitNumber_FreqMap[ECALDCC_OrbitNumber];
	      ECALDCC_OrbitNumber_MostFreqId = ECALDCC_OrbitNumber;
	    }
	    
	    if ( ECALDCC_BunchCrossing_FreqMap[ECALDCC_BunchCrossing] > ECALDCC_BunchCrossing_MostFreqCounts ) {
	      ECALDCC_BunchCrossing_MostFreqCounts = ECALDCC_BunchCrossing_FreqMap[ECALDCC_BunchCrossing];
	      ECALDCC_BunchCrossing_MostFreqId = ECALDCC_BunchCrossing;
	    }
	    
	    if ( ECALDCC_TriggerType_FreqMap[ECALDCC_TriggerType] > ECALDCC_TriggerType_MostFreqCounts ) {
	      ECALDCC_TriggerType_MostFreqCounts = ECALDCC_TriggerType_FreqMap[ECALDCC_TriggerType];
	      ECALDCC_TriggerType_MostFreqId = ECALDCC_TriggerType;
	    }

	  }
	  
	}
	
      } else {
	LogWarning("EBRawDataTask") << EcalRawDataCollection_ << " not available";
      }
      
    }
    
    // ECAL barrel FEDs
    int EBFirstFED=610;
    for(int i=0; i<36; i++) {
      
      const FEDRawData& fedData = allFedRawData->FEDData(EBFirstFED+i);
      
      int length = fedData.size()/sizeof(uint64_t);

      if ( length > 0 ) {
	
	uint64_t * pData = (uint64_t *)(fedData.data());
	uint64_t * fedTrailer = pData + (length - 1);
	bool crcError = (*fedTrailer >> 2 ) & 0x1; 
	
	if (crcError) meEBCRCErrors_->Fill( i+1 );
	
      }
      
    }
    

  } else {
    LogWarning("EBRawDataTask") << FEDRawDataCollection_ << " not available";
  }

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {
    
    if ( dcchs.isValid() ) {
      
      for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

	EcalDCCHeaderBlock dcch = (*dcchItr);
	
	if ( Numbers::subDet( dcch ) != EcalBarrel ) continue;
	
	int ism = Numbers::iSM( dcch, EcalBarrel );
	float xism = ism+0.5;

	int ECALDCC_L1A = dcch.getLV1();
	int ECALDCC_OrbitNumber = dcch.getOrbit();
	int ECALDCC_BunchCrossing = dcch.getBX();
	int ECALDCC_TriggerType = dcch.getBasicTriggerType();

	if ( gtFedDataSize > 0 ) {
	  
	  if ( GT_L1A != ECALDCC_L1A ) meEBL1AErrors_->Fill( xism );
	  
	  if ( GT_BunchCrossing != ECALDCC_BunchCrossing ) meEBBunchCrossingErrors_->Fill( xism );
	  
	  if ( GT_TriggerType != ECALDCC_TriggerType ) meEBTriggerTypeErrors_->Fill ( xism );

	} else {

	  if ( ECALDCC_L1A_MostFreqId != ECALDCC_L1A ) meEBL1AErrors_->Fill( xism );
	  
	  if ( ECALDCC_BunchCrossing_MostFreqId != ECALDCC_BunchCrossing ) meEBBunchCrossingErrors_->Fill( xism );
	  
	  if ( ECALDCC_TriggerType_MostFreqId != ECALDCC_TriggerType ) meEBTriggerTypeErrors_->Fill ( xism );

	}

	if ( GT_OrbitNumber_Present ) {

	  if ( GT_OrbitNumber != ECALDCC_OrbitNumber ) meEBOrbitNumberErrors_->Fill ( xism );

	} else {
	  
	  if ( ECALDCC_OrbitNumber_MostFreqId != ECALDCC_OrbitNumber ) meEBOrbitNumberErrors_->Fill ( xism );
	  
	}

      }
      
    }

  } else {
    LogWarning("EBRawDataTask") << EcalRawDataCollection_ << " not available";
  }

}

