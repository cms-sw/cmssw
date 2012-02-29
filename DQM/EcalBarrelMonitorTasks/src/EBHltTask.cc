/*
 * \file EBHltTask.cc
 *
 * $Date: 2012/02/28 16:38:13 $
 * $Revision: 1.18 $
 * \author G. Della Ricca
 *
*/

#include <iostream>
#include <fstream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBHltTask.h"

EBHltTask::EBHltTask(const edm::ParameterSet& ps){

  init_ = false;

  initGeometry_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  // workaround Feb 29 2012
  bool forcePrefixME(ps.getUntrackedParameter<bool>("forcePrefixME", false));

  prefixME_ = forcePrefixME ? ps.getUntrackedParameter<std::string>("prefixME", "Ecal") : "Ecal";

  folderName_ = ps.getUntrackedParameter<std::string>("folderName", "FEDIntegrity");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EBDetIdCollection1_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection1");
  EBDetIdCollection2_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection2");
  EBDetIdCollection3_ =  ps.getParameter<edm::InputTag>("EBDetIdCollection3");
  EcalElectronicsIdCollection1_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection1");
  EcalElectronicsIdCollection2_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection2");
  EcalElectronicsIdCollection3_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection3");
  EcalElectronicsIdCollection4_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection4");
  EcalElectronicsIdCollection5_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection5");
  EcalElectronicsIdCollection6_ = ps.getParameter<edm::InputTag>("EcalElectronicsIdCollection6");
  FEDRawDataCollection_ = ps.getParameter<edm::InputTag>("FEDRawDataCollection");

  meEBFedsOccupancy_ = 0;
  meEBFedsSizeErrors_ = 0;
  meEBFedsIntegrityErrors_ = 0;

  map = 0;

  ievt_ = 0;

}

EBHltTask::~EBHltTask(){

}

void EBHltTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/" + folderName_);
    dqmStore_->rmdir(prefixME_ + "/" + folderName_);
  }

}

void EBHltTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  initGeometry(c);

  if ( ! mergeRuns_ ) this->reset();

}

void EBHltTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBHltTask::reset(void) {

  if ( meEBFedsOccupancy_ ) meEBFedsOccupancy_->Reset();
  if ( meEBFedsSizeErrors_ ) meEBFedsSizeErrors_->Reset();
  if ( meEBFedsIntegrityErrors_ ) meEBFedsIntegrityErrors_->Reset();

}

void EBHltTask::setup(void){

  init_ = true;

  std::string name;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/" + folderName_);

    name = "FEDEntries";
    meEBFedsOccupancy_ = dqmStore_->book1D(name, name, 54, 601, 655);

    name = "FEDFatal";
    meEBFedsSizeErrors_ = dqmStore_->book1D(name, name, 54, 601, 655);

    name = "FEDNonFatal";
    meEBFedsIntegrityErrors_ = dqmStore_->book1D(name, name, 54, 601, 655);

  }

}

void EBHltTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/" + folderName_);

    if ( meEBFedsOccupancy_ ) dqmStore_->removeElement( meEBFedsOccupancy_->getName() );
    meEBFedsOccupancy_ = 0;

    if ( meEBFedsSizeErrors_ ) dqmStore_->removeElement( meEBFedsSizeErrors_->getName() );
    meEBFedsSizeErrors_ = 0;

    if ( meEBFedsIntegrityErrors_ ) dqmStore_->removeElement( meEBFedsIntegrityErrors_->getName() );
    meEBFedsIntegrityErrors_ = 0;

  }

  init_ = false;

}

void EBHltTask::endJob(void){

  edm::LogInfo("EBHltTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBHltTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  int EcalFirstFED = 601;
  int EBFirstFED = 610;

  int FedsSizeErrors[54];
  for ( int i=0; i<54; i++ ) FedsSizeErrors[i]=0;

  edm::Handle<FEDRawDataCollection> allFedRawData;

  if ( e.getByLabel(FEDRawDataCollection_, allFedRawData) ) {

    for ( int ism=1; ism<=54; ism++ ) {

      const FEDRawData& fedData = allFedRawData->FEDData( EcalFirstFED + ism - 1 );

      int length = fedData.size()/sizeof(uint64_t);

      if ( length > 0 ) {

	if ( meEBFedsOccupancy_ ) meEBFedsOccupancy_->Fill( EcalFirstFED + ism - 1 );

	uint64_t * pData = (uint64_t *)(fedData.data());
	uint64_t * fedTrailer = pData + (length - 1);
	bool crcError = (*fedTrailer >> 2 ) & 0x1;

	if (crcError) FedsSizeErrors[ism-1]++;

      }

    }

  } else {
    edm::LogWarning("EBHltTask") << FEDRawDataCollection_ << " not available";
  }


  for( int ism=1; ism<=54; ism++ ) {

    if ( FedsSizeErrors[ism-1] != 0 ) {

      if ( meEBFedsSizeErrors_ ) meEBFedsSizeErrors_->Fill( EcalFirstFED + ism - 1 );

    }

  }


  // Integrity errors
  edm::Handle<EBDetIdCollection> ebIds1;

  if ( e.getByLabel(EBDetIdCollection1_, ebIds1) ) {

    for ( EBDetIdCollection::const_iterator idItr = ebIds1->begin(); idItr != ebIds1->end(); ++idItr ) {

      int ism = iSM( *idItr );

      if( ism > -1 ) meEBFedsIntegrityErrors_->Fill( EBFirstFED + ism - 1, 1./1700.);

    }

  } else {

    edm::LogWarning("EBHltTask") << EBDetIdCollection1_ << " not available";

  }

  edm::Handle<EEDetIdCollection> eeIds1;

  if ( e.getByLabel(EBDetIdCollection1_, eeIds1) ) {

    for ( EEDetIdCollection::const_iterator idItr = eeIds1->begin(); idItr != eeIds1->end(); ++idItr ) {

      int ism = iSM( *idItr );
      int fednumber = ( ism < 10 ) ? 600 + ism : 636 + ism;

      if ( ism > -1 ) meEBFedsIntegrityErrors_->Fill( fednumber, 1./850.);

    }

  } else {

    edm::LogWarning("EEHltTask") << EBDetIdCollection1_ << " not available";

  }

  edm::Handle<EBDetIdCollection> ebIds2;

  if ( e.getByLabel(EBDetIdCollection2_, ebIds2) ) {

    for ( EBDetIdCollection::const_iterator idItr = ebIds2->begin(); idItr != ebIds2->end(); ++idItr ) {

      int ism = iSM( *idItr );

      if ( ism > -1 ) meEBFedsIntegrityErrors_->Fill( EBFirstFED + ism - 1, 1./1700.);

    }

  } else {

    edm::LogWarning("EBHltTask") << EBDetIdCollection2_ << " not available";

  }

  edm::Handle<EEDetIdCollection> eeIds2;

  if ( e.getByLabel(EBDetIdCollection2_, eeIds2) ) {

    for ( EEDetIdCollection::const_iterator idItr = eeIds2->begin(); idItr != eeIds2->end(); ++idItr ) {

      int ism = iSM( *idItr );
      int fednumber = ( ism < 10 ) ? 600 + ism : 636 + ism;

      if ( ism > -1 ) meEBFedsIntegrityErrors_->Fill( fednumber, 1./850.);

    }

  } else {

    edm::LogWarning("EEHltTask") << EBDetIdCollection2_ << " not available";

  }


  edm::Handle<EBDetIdCollection> ebIds3;

  if ( e.getByLabel(EBDetIdCollection3_, ebIds3) ) {

    for ( EBDetIdCollection::const_iterator idItr = ebIds3->begin(); idItr != ebIds3->end(); ++idItr ) {

      int ism = iSM( *idItr );

      if ( ism > -1 ) meEBFedsIntegrityErrors_->Fill( EcalFirstFED + ism - 1, 1./1700.);

    }

  } else {

    edm::LogWarning("EBHltTask") << EBDetIdCollection3_ << " not available";

  }

  edm::Handle<EEDetIdCollection> eeIds3;

  if ( e.getByLabel(EBDetIdCollection3_, eeIds3) ) {

    for ( EEDetIdCollection::const_iterator idItr = eeIds3->begin(); idItr != eeIds3->end(); ++idItr ) {

      int ism = iSM( *idItr );
      int fednumber = ( ism < 10 ) ? 600 + ism : 636 + ism;

      if ( ism > -1 ) meEBFedsIntegrityErrors_->Fill( fednumber, 1./850.);

    }

  } else {

    edm::LogWarning("EEHltTask") << EBDetIdCollection3_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids4;

  if ( e.getByLabel(EcalElectronicsIdCollection1_, ids4) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids4->begin(); idItr != ids4->end(); ++idItr ) {

      int idcc(idItr->dccId());

      float weight;

      if(idcc >= 10 && idcc <= 45) weight = 1./68.;
      else weight = 1./34.;

      meEBFedsIntegrityErrors_->Fill( EcalFirstFED + idcc - 1, weight);

    }

  } else {

    edm::LogWarning("EBHltTask") << EcalElectronicsIdCollection1_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids5;

  if ( e.getByLabel(EcalElectronicsIdCollection2_, ids5) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids5->begin(); idItr != ids5->end(); ++idItr ) {

      int idcc(idItr->dccId());

      float weight;

      if(idcc >= 10 && idcc <= 45) weight = 1./68.;
      else weight = 1./34.;

      meEBFedsIntegrityErrors_->Fill( EcalFirstFED + idcc - 1, weight);

    }

  } else {

    edm::LogWarning("EBHltTask") << EcalElectronicsIdCollection2_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids6;

  if ( e.getByLabel(EcalElectronicsIdCollection3_, ids6) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids6->begin(); idItr != ids6->end(); ++idItr ) {

      int idcc(idItr->dccId());

      float weight;

      if(idcc >= 10 && idcc <= 45) weight = 1./68.;
      else weight = 1./34.;

      meEBFedsIntegrityErrors_->Fill( EcalFirstFED + idcc - 1, weight);

    }

  } else {

    edm::LogWarning("EBHltTask") << EcalElectronicsIdCollection3_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids7;

  if ( e.getByLabel(EcalElectronicsIdCollection4_, ids7) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids7->begin(); idItr != ids7->end(); ++idItr ) {

      int idcc(idItr->dccId());

      float weight;

      if(idcc >= 10 && idcc <= 45) weight = 1./68.;
      else weight = 1./34.;

      meEBFedsIntegrityErrors_->Fill( EcalFirstFED + idcc - 1, weight);

    }

  } else {

    edm::LogWarning("EBHltTask") << EcalElectronicsIdCollection4_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids8;

  if ( e.getByLabel(EcalElectronicsIdCollection5_, ids8) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids8->begin(); idItr != ids8->end(); ++idItr ) {

      int idcc(idItr->dccId());

      float weight;

      if(idcc >= 10 && idcc <= 45) weight = 1./68.;
      else weight = 1./34.;

      meEBFedsIntegrityErrors_->Fill( EcalFirstFED + idcc - 1, weight);

    }

  } else {

    edm::LogWarning("EBHltTask") << EcalElectronicsIdCollection5_ << " not available";

  }

  edm::Handle<EcalElectronicsIdCollection> ids9;

  if ( e.getByLabel(EcalElectronicsIdCollection6_, ids9) ) {

    for ( EcalElectronicsIdCollection::const_iterator idItr = ids9->begin(); idItr != ids9->end(); ++idItr ) {

      int idcc(idItr->dccId());

      float weight;

      if(idcc >= 10 && idcc <= 45) weight = 1./68.;
      else weight = 1./34.;

      meEBFedsIntegrityErrors_->Fill( EcalFirstFED + idcc - 1, weight);

    }

  } else {

    edm::LogWarning("EBHltTask") << EcalElectronicsIdCollection6_ << " not available";

  }

}

//-------------------------------------------------------------------------

void EBHltTask::initGeometry( const edm::EventSetup& setup ) {

  if( initGeometry_ ) return;

  initGeometry_ = true;

  edm::ESHandle< EcalElectronicsMapping > handle;
  setup.get< EcalMappingRcd >().get(handle);
  map = handle.product();

  if( ! map ) edm::LogWarning("EBHltTask") << "EcalElectronicsMapping not available";

}

int EBHltTask::iSM( const EBDetId& id ) {

  if( ! map ) return -1;

  EcalElectronicsId eid = map->getElectronicsId(id);
  int idcc = eid.dccId();

  // EB-/EB+
  if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

  edm::LogWarning("EBHltTask") << "Wrong DCC id: dcc = " << idcc;
  return -1;

}

int EBHltTask::iSM( const EcalElectronicsId& id ) {

  int idcc = id.dccId();

  // EB-/EB+
  if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

  edm::LogWarning("EBHltTask") << "Wrong DCC id: dcc = " << idcc;
  return -1;

}

