/*
 * \file EEHltTask.cc
 *
 * $Date: 2008/12/03 10:28:11 $
 * $Revision: 1.8 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EEHltTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EEHltTask::EEHltTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EEDetIdCollection0_ =  ps.getParameter<edm::InputTag>("EEDetIdCollection0");
  EEDetIdCollection1_ =  ps.getParameter<edm::InputTag>("EEDetIdCollection1");
  EEDetIdCollection2_ =  ps.getParameter<edm::InputTag>("EEDetIdCollection2");
  EEDetIdCollection3_ =  ps.getParameter<edm::InputTag>("EEDetIdCollection3");
  EcalElectronicsIdCollection1_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection1");
  EcalElectronicsIdCollection2_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection2");
  EcalElectronicsIdCollection3_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection3");
  EcalElectronicsIdCollection4_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection4");
  EcalElectronicsIdCollection5_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection5");
  EcalElectronicsIdCollection6_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection6");
  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");

  meEEFedsOccupancy_ = 0;
  meEEFedsSizeErrors_ = 0;
  meEEFedsIntegrityErrors_ = 0;
  
}

EEHltTask::~EEHltTask(){

}

void EEHltTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/FEDIntegrity");
    dqmStore_->rmdir(prefixME_ + "/FEDIntegrity");
  }

  Numbers::initGeometry(c, false);

}

void EEHltTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EEHltTask::endRun(const Run& r, const EventSetup& c) {

}

void EEHltTask::reset(void) {

  if ( meEEFedsOccupancy_ ) meEEFedsOccupancy_->Reset();
  if ( meEEFedsSizeErrors_ ) meEEFedsSizeErrors_->Reset();
  if ( meEEFedsIntegrityErrors_ ) meEEFedsIntegrityErrors_->Reset();

}

void EEHltTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/FEDIntegrity");

    sprintf(histo, "FEDEntries");
    meEEFedsOccupancy_ = dqmStore_->book1D(histo, histo, 54, 601, 655);

    sprintf(histo, "FEDFatal");
    meEEFedsSizeErrors_ = dqmStore_->book1D(histo, histo, 54, 601, 655);

    sprintf(histo, "FEDNonFatal");
    meEEFedsIntegrityErrors_ = dqmStore_->book1D(histo, histo, 54, 601, 655);

  }

}

void EEHltTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/FEDIntegrity");

    if ( meEEFedsOccupancy_ ) dqmStore_->removeElement( meEEFedsOccupancy_->getName() );
    meEEFedsOccupancy_ = 0;

    if ( meEEFedsSizeErrors_ ) dqmStore_->removeElement( meEEFedsSizeErrors_->getName() );
    meEEFedsSizeErrors_ = 0;

    if ( meEEFedsIntegrityErrors_ ) dqmStore_->removeElement( meEEFedsIntegrityErrors_->getName() );
    meEEFedsIntegrityErrors_ = 0;

  }

  init_ = false;

}

void EEHltTask::endJob(void){

  LogInfo("EEHltTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EEHltTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // ECAL endcap FEDs
  int EEFirstFED[2];
  EEFirstFED[0] = 601; // EE-
  EEFirstFED[1] = 646; // EE+

  int FedsSizeErrors[18];
  for ( int i=0; i<18; i++ ) FedsSizeErrors[i]=0;

  edm::Handle<EEDetIdCollection> ids0;

  if ( e.getByLabel(EEDetIdCollection0_, ids0) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids0->begin(); idItr != ids0->end(); ++idItr ) {

      int ism = Numbers::iSM( *idItr );

      FedsSizeErrors[ism-1]++;

    }

  } else {

//    LogWarning("EEHltTask") << EEDetIdCollection0_ << " not available";

  }


  edm::Handle<FEDRawDataCollection> allFedRawData;

  if ( e.getByLabel(FEDRawDataCollection_, allFedRawData) ) {

    for(int zside=0; zside<2; zside++) {
      
      int firstFedOnSide=EEFirstFED[zside];
      
      for ( int ism=1; ism<=9; ism++ ) {
	
	const FEDRawData& fedData = allFedRawData->FEDData( firstFedOnSide + ism - 1 );
	
	int length = fedData.size()/sizeof(uint64_t);
	
	if ( length > 0 ) {
	  
	  if ( meEEFedsOccupancy_ ) meEEFedsOccupancy_->Fill( firstFedOnSide + ism - 1 );
	  
	  uint64_t * pData = (uint64_t *)(fedData.data());
	  uint64_t * fedTrailer = pData + (length - 1);
	  bool crcError = (*fedTrailer >> 2 ) & 0x1;
	  
	  if (crcError) FedsSizeErrors[ism-1]++;
	  
	}
	
      }
      
    }

  } else {
    LogWarning("EEHltTask") << FEDRawDataCollection_ << " not available";
  }


  for( int ism=1; ism<=18; ism++ ) {

    if ( FedsSizeErrors[ism-1] != 0 ) {

      int fednumber = ( ism < 10 ) ? 600 + ism : 636 + ism;

      if ( meEEFedsSizeErrors_ ) meEEFedsSizeErrors_->Fill( fednumber );

    }

  }


  // Integrity errors
  Handle<EEDetIdCollection> ids1;

  if ( e.getByLabel(EEDetIdCollection1_, ids1) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids1->begin(); idItr != ids1->end(); ++idItr ) {

      int ism = Numbers::iSM( *idItr );
      int fednumber = ( ism < 10 ) ? 600 + ism : 636 + ism;
      
      meEEFedsIntegrityErrors_->Fill( fednumber, 1./850.);

    }

  } else {

    LogWarning("EEHltTask") << EEDetIdCollection1_ << " not available";

  }

  Handle<EEDetIdCollection> ids2;

  if ( e.getByLabel(EEDetIdCollection2_, ids2) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids2->begin(); idItr != ids2->end(); ++idItr ) {

      int ism = Numbers::iSM( *idItr );
      int fednumber = ( ism < 10 ) ? 600 + ism : 636 + ism;

      meEEFedsIntegrityErrors_->Fill( fednumber, 1./850.);

    }

  } else {

    LogWarning("EEHltTask") << EEDetIdCollection2_ << " not available";

  }

  Handle<EEDetIdCollection> ids3;

  if ( e.getByLabel(EEDetIdCollection3_, ids3) ) {

    for ( EEDetIdCollection::const_iterator idItr = ids3->begin(); idItr != ids3->end(); ++idItr ) {

      int ism = Numbers::iSM( *idItr );
      int fednumber = ( ism < 10 ) ? 600 + ism : 636 + ism;

      meEEFedsIntegrityErrors_->Fill( fednumber, 1./850.);

    }

  } else {

    LogWarning("EEHltTask") << EEDetIdCollection3_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids4;

  if ( e.getByLabel(EcalElectronicsIdCollection1_, ids4) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids4->begin(); idItr != ids4->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ismt = Numbers::iSM( *idItr );
      int fednumber = ( ismt < 10 ) ? 600 + ismt : 636 + ismt;

      meEEFedsIntegrityErrors_->Fill( fednumber, 1./34.);

    }

  } else {

    LogWarning("EEHltTask") << EcalElectronicsIdCollection1_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids5;

  if ( e.getByLabel(EcalElectronicsIdCollection2_, ids5) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids5->begin(); idItr != ids5->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *idItr );
      int fednumber = ( ism < 10 ) ? 600 + ism : 636 + ism;

      meEEFedsIntegrityErrors_->Fill( fednumber, 1./850.);

    }

  } else {

    LogWarning("EEHltTask") << EcalElectronicsIdCollection2_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids6;

  if ( e.getByLabel(EcalElectronicsIdCollection3_, ids6) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids6->begin(); idItr != ids6->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ismt = Numbers::iSM( *idItr );
      int fednumber = ( ismt < 10 ) ? 600 + ismt : 636 + ismt;

      meEEFedsIntegrityErrors_->Fill( fednumber, 1./34.);

    }

  } else {

    LogWarning("EEHltTask") << EcalElectronicsIdCollection3_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids7;

  if ( e.getByLabel(EcalElectronicsIdCollection4_, ids7) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids7->begin(); idItr != ids7->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *idItr );
      int fednumber = ( ism < 10 ) ? 600 + ism : 636 + ism;

      meEEFedsIntegrityErrors_->Fill( fednumber, 1./850.);

    }

  } else {

    LogWarning("EEHltTask") << EcalElectronicsIdCollection4_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids8;

  if ( e.getByLabel(EcalElectronicsIdCollection5_, ids8) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids8->begin(); idItr != ids8->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *idItr );
      int fednumber = ( ism < 10 ) ? 600 + ism : 636 + ism;

      meEEFedsIntegrityErrors_->Fill( fednumber, 1./850.);

    }

  } else {

    LogWarning("EEHltTask") << EcalElectronicsIdCollection5_ << " not available";

  }

  Handle<EcalElectronicsIdCollection> ids9;

  if ( e.getByLabel(EcalElectronicsIdCollection6_, ids9) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids9->begin(); idItr != ids9->end(); ++idItr ) {

      if ( Numbers::subDet( *idItr ) != EcalEndcap ) continue;

      int ism = Numbers::iSM( *idItr );
      int fednumber = ( ism < 10 ) ? 600 + ism : 636 + ism;

      meEEFedsIntegrityErrors_->Fill( fednumber, 1./850.);

    }

  } else {

    LogWarning("EEHltTask") << EcalElectronicsIdCollection6_ << " not available";

  }

}

