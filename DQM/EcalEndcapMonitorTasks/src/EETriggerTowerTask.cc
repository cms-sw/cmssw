/*
 * \file EETriggerTowerTask.cc
 *
 * $Date: 2007/11/14 11:18:07 $
 * $Revision: 1.16 $
 * \author C. Bernet
 * \author G. Della Ricca
 * \author E. Di Marco
 *
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EETriggerTowerTask.h"

using namespace cms;
using namespace edm;
using namespace std;


const int EETriggerTowerTask::nTTEta = 20;
const int EETriggerTowerTask::nTTPhi = 20;
const int EETriggerTowerTask::nSM = 18;


EETriggerTowerTask::EETriggerTowerTask(const ParameterSet& ps) {

  init_ = false;

  reserveArray(meEtMapReal_);
  reserveArray(meVetoReal_);
  reserveArray(meFlagsReal_);
  reserveArray(meEtMapEmul_);
  reserveArray(meVetoEmul_);
  reserveArray(meFlagsEmul_);
  reserveArray(meEmulError_);
  reserveArray(meVetoEmulError_);
  reserveArray(meFlagEmulError_);

  realCollection_ =  ps.getParameter<InputTag>("EcalTrigPrimDigiCollectionReal");
  emulCollection_ =  ps.getParameter<InputTag>("EcalTrigPrimDigiCollectionEmul");

//   realModuleLabel_
//     = ps.getUntrackedParameter<string>("real_digis_moduleLabel",
// 				       "ecalEBunpacker");
//   emulModuleLabel_
//     = ps.getUntrackedParameter<string>("emulated_digis_moduleLabel",
// 				       "ecalTriggerPrimitiveDigis");
  outputFile_
    = ps.getUntrackedParameter<string>("OutputRootFile",
				       "");


  ostringstream  str;
  str<<"Module label for producer of REAL     digis: "<<realCollection_<<endl;
  str<<"Module label for producer of EMULATED digis: "<<emulCollection_<<endl;

  LogDebug("EETriggerTowerTask")<<str.str()<<endl;
}


EETriggerTowerTask::~EETriggerTowerTask(){

}


void EETriggerTowerTask::reserveArray( array1& array ) {

  array.reserve( nSM );
  array.resize( nSM, static_cast<MonitorElement*>(0) );

}

void EETriggerTowerTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalEndcap/EETriggerTowerTask");
    dbe->rmdir("EcalEndcap/EETriggerTowerTask");
  }

}


void EETriggerTowerTask::setup(void){

  init_ = true;


//   DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  DaqMonitorBEInterface* dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    // dbe->showDirStructure();

    setup( dbe,
	   "Real Digis",
	   "EcalEndcap/EETriggerTowerTask", false );

    setup( dbe,
	   "Emulated Digis",
	   "EcalEndcap/EETriggerTowerTask/Emulated", true);
  }
  else {
    LogError("EETriggerTowerTask")<<"Bad DaqMonitorBEInterface, "
				  <<"cannot book MonitorElements."<<endl;
  }
}


void EETriggerTowerTask::setup( DaqMonitorBEInterface* dbe,
				const char* nameext,
				const char* folder,
				bool emulated ) {


  array1*  meEtMap = &meEtMapReal_;
  array1*  meVeto = &meVetoReal_;
  array1*  meFlags = &meFlagsReal_;

  if( emulated ) {
    meEtMap = &meEtMapEmul_;
    meVeto = &meVetoEmul_;
    meFlags= &meFlagsEmul_;
  }


  assert(dbe);

  dbe->setCurrentFolder(folder);


  static const unsigned namesize = 200;

  char histo[namesize];
  sprintf(histo, "EETTT Et map %s", nameext);
  string etMapName = histo;
  sprintf(histo, "EETTT FineGrainVeto %s", nameext);
  string fineGrainVetoName = histo;
  sprintf(histo, "EETTT Flags %s", nameext);
  string flagsName = histo;
  string emulErrorName = "EETTT EmulError";
  string emulFineGrainVetoErrorName = "EETTT EmulFineGrainVetoError";
  string emulFlagErrorName = "EETTT EmulFlagError";

  for (int i = 0; i < 18 ; i++) {

    string etMapNameSM = etMapName;
    etMapNameSM += " " + Numbers::sEE(i+1);

    (*meEtMap)[i] = dbe->book3D(etMapNameSM.c_str(), etMapNameSM.c_str(),
				50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
				50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
				256, 0, 256.);
    (*meEtMap)[i]->setAxisTitle("ix", 1);
    (*meEtMap)[i]->setAxisTitle("iy", 2);
    dbe->tag((*meEtMap)[i], i+1);

    string  fineGrainVetoNameSM = fineGrainVetoName;
    fineGrainVetoNameSM += " " + Numbers::sEE(i+1);

    (*meVeto)[i] = dbe->book3D(fineGrainVetoNameSM.c_str(),
			       fineGrainVetoNameSM.c_str(),
			       50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
			       50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
			       2, 0., 2.);
    (*meVeto)[i]->setAxisTitle("ix", 1);
    (*meVeto)[i]->setAxisTitle("iy", 2);
    dbe->tag((*meVeto)[i], i+1);

    string  flagsNameSM = flagsName;
    flagsNameSM += " " + Numbers::sEE(i+1);

    (*meFlags)[i] = dbe->book3D(flagsNameSM.c_str(), flagsNameSM.c_str(),
				50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
				50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
				8, 0., 8.);
    (*meFlags)[i]->setAxisTitle("ix", 1);
    (*meFlags)[i]->setAxisTitle("iy", 2);
    dbe->tag((*meFlags)[i], i+1);


    if(!emulated) {

      string  emulErrorNameSM = emulErrorName;
      emulErrorNameSM += " " + Numbers::sEE(i+1);

      meEmulError_[i] = dbe->book2D(emulErrorNameSM.c_str(),
				    emulErrorNameSM.c_str(),
				    50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
				    50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50. );
      meEmulError_[i]->setAxisTitle("ix", 1);
      meEmulError_[i]->setAxisTitle("iy", 2);
      dbe->tag(meEmulError_[i], i+1);

      string  emulFineGrainVetoErrorNameSM = emulFineGrainVetoErrorName;
      emulFineGrainVetoErrorNameSM += " " + Numbers::sEE(i+1);

      meVetoEmulError_[i] = dbe->book3D(emulFineGrainVetoErrorNameSM.c_str(),
					  emulFineGrainVetoErrorNameSM.c_str(),
					  50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
					  50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
					  8, 0., 8.);
      meVetoEmulError_[i]->setAxisTitle("ix", 1);
      meVetoEmulError_[i]->setAxisTitle("iy", 2);
      dbe->tag(meVetoEmulError_[i], i+1);

      string  emulFlagErrorNameSM = emulFlagErrorName;
      emulFlagErrorNameSM += " " + Numbers::sEE(i+1);

      meFlagEmulError_[i] = dbe->book3D(emulFlagErrorNameSM.c_str(),
					  emulFlagErrorNameSM.c_str(),
					  50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50.,
					  50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.,
					  8, 0., 8.);
      meFlagEmulError_[i]->setAxisTitle("ix", 1);
      meFlagEmulError_[i]->setAxisTitle("iy", 2);
      dbe->tag(meFlagEmulError_[i], i+1);

    }
  }

}


void EETriggerTowerTask::cleanup(void) {

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {

    if( !outputFile_.empty() )
      dbe->save( outputFile_.c_str() );

    dbe->rmdir( "EcalEndcap/EETriggerTowerTask" );
  }

  init_ = false;

}




void EETriggerTowerTask::endJob(void){

  LogInfo("EETriggerTowerTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();
}


void EETriggerTowerTask::analyze(const Event& e, const EventSetup& c){

  Numbers::initGeometry(c);

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalTrigPrimDigiCollection> realDigis;

  if ( e.getByLabel(realCollection_, realDigis) ) {

    int neetpd = realDigis->size();
    LogDebug("EETriggerTowerTask")
      <<"event "
      <<ievt_
      <<" trigger primitive digi collection size: "
      <<neetpd;

    processDigis( realDigis,
		  meEtMapReal_,
		  meVetoReal_,
		  meFlagsReal_);

  } else {
    LogWarning("EBTriggerTowerTask")
      << realCollection_ << " not available"; 
  }

  Handle<EcalTrigPrimDigiCollection> emulDigis;

  if ( e.getByLabel(emulCollection_, emulDigis) ) {

    processDigis( emulDigis,
		  meEtMapEmul_,
		  meVetoEmul_,
		  meFlagsEmul_,
		  realDigis);


  } else {
    LogWarning("EETriggerTowerTask")
      << emulCollection_ << " not available";
  }

}

void
EETriggerTowerTask::processDigis( const Handle<EcalTrigPrimDigiCollection>&
				  digis,
				  array1& meEtMap,
				  array1& meVeto,
				  array1& meFlags,
				  const Handle<EcalTrigPrimDigiCollection>&
				  compDigis ) {

  LogDebug("EETriggerTowerTask")<<"processing "<<meEtMap[0]->getName()<<endl;

  ostringstream  str;
  typedef EcalTrigPrimDigiCollection::const_iterator ID;
  for ( ID tpdigiItr = digis->begin();
	tpdigiItr != digis->end(); ++tpdigiItr ) {

    EcalTriggerPrimitiveDigi data = (*tpdigiItr);
    EcalTrigTowerDetId idt = data.id();

    if ( idt.subDet() != EcalEndcap ) continue;

    int ismt = Numbers::iSM( idt );

    int itt = Numbers::iTT( idt );

    vector<DetId> crystals = Numbers::crystals( idt );

    for ( unsigned int i=0; i<crystals.size(); i++ ) {

    EEDetId id = crystals[i];

    int ix = id.ix();
    int iy = id.iy();

    if ( ismt >= 1 && ismt <= 9 ) ix = 101 - ix;

    float xix = ix+0.5;
    float xiy = iy+0.5;

    str<<"det id = "<<id.rawId()<<" "
       <<id<<" sm, tt, x, y "<<ismt<<" "<<itt<<" "<<ix<<" "<<iy<<endl;

    float xval;

    xval = data.compressedEt();
    if ( meEtMap[ismt-1] ) {
      meEtMap[ismt-1]->Fill(xix-1, xiy-1, xval);
    }
    else {
      LogError("EETriggerTowerTask")<<"histo does not exist "<<endl;
    }

    xval = 0.5 + data.fineGrain();
    if ( meVeto[ismt-1] ) meVeto[ismt-1]->Fill(xix-1, xiy-1, xval);

    xval = 0.5 + data.ttFlag();
    if ( meFlags[ismt-1] ) meFlags[ismt-1]->Fill(xix-1, xiy-1, xval);


    if( compDigis.isValid() ) {
      ID compDigiItr = compDigis->find( idt.rawId() );

      bool good = true;
      bool goodFlag = true;
      bool goodVeto = true;
      if( compDigiItr != compDigis->end() ) {
	str<<"found corresponding digi! "<<*compDigiItr<<endl;
	if( data.compressedEt() != compDigiItr->compressedEt() ) {
	  str<<"but it is different..."<<endl;
	  good = false;
	}
	if( data.ttFlag() != compDigiItr->ttFlag() ) {
	  str<<"but flag is different..."<<endl;
	  goodFlag = false;
	}
	if( data.fineGrain() != compDigiItr->fineGrain() ) {
	  str<<"but fine grain veto is different..."<<endl;
	  goodVeto = false;
	}
      }
      else {
	good = false;
	goodFlag = false;
	goodVeto = false;
	str<<"could not find corresponding digi... "<<endl;
      }
      if(!good ) {
	if ( meEmulError_[ismt-1] ) meEmulError_[ismt-1]->Fill(xix-1, xiy-1);
      }
      if(!goodFlag) {
	float zval = data.ttFlag();
	if ( meFlagEmulError_[ismt-1] ) meFlagEmulError_[ismt-1]->Fill(xix-1, xiy-1, zval);
      }
      if(!goodVeto) {
	float zval = data.fineGrain();
	if ( meVetoEmulError_[ismt-1] ) meVetoEmulError_[ismt-1]->Fill(xix-1, xiy-1, zval);
      }
    }
  }
  }
  LogDebug("EETriggerTowerTask")<<str.str()<<endl;
}
