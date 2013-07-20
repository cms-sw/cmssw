// -*- C++ -*-
//
// Package:    ScalersRecover
// Class:      ScalersRecover
// 
/**\class ScalersRecover ScalersRecover.cc DataFormats/ScalersRecover/src/ScalersRecover.cc

 Description: EDAnalyzer to fetch trigger scalers and convert it to SQL

 Implementation:
   This module may be used to recover L1 trigger scalers data from 
   CMS raw data files, and convert it to a series of SQL INSERT statements 
   that can be used to back-populate the corresponding L1 database 
   tables.   Should be performed on a run-by-run basis as necessary.
   First data file processed MUST contain data from the first lumi 
   sections.  In general, the files should be in lumi section order.

   We recommend running the job on MinBias RECO files.   If you 
   run with RAW files, you will have to include the ScalerRawToDigi 
   conversion module.

   The resulting SQL commands will be contained in a file 
   named scalers.sql
*/
//
// Original Author:  William Badgett
//         Created:  Mon May 24 14:45:17 CEST 2010
// $Id: ScalersRecover.cc,v 1.4 2010/05/25 16:38:51 badgett Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include <ctime>
#include "DataFormats/Scalers/interface/TimeSpec.h"
//
// class declaration
//

class ScalersRecover : public edm::EDAnalyzer 
{
  public:
    explicit ScalersRecover(const edm::ParameterSet&);
    ~ScalersRecover();


  private:
    virtual void beginJob() ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;

    int lastLumiSection;
    FILE * sql;
};

// Constructor
ScalersRecover::ScalersRecover(const edm::ParameterSet& iConfig)

{
  sql = NULL;
  sql = fopen("scalers.sql","w");
}

// Destructor
ScalersRecover::~ScalersRecover()
{
  if ( sql != NULL ) { fclose(sql);}
}

// ------------ method called to for each event  ------------
void ScalersRecover::analyze(const edm::Event& iEvent, 
			     const edm::EventSetup& iSetup)
{
  using namespace edm;
  char heure [32];
  char sNanos [16];
  struct tm *hora;

  edm::Handle<Level1TriggerScalersCollection> data;
  bool ok = iEvent.getByLabel("scalersRawToDigi",data);

  if ( !ok ) 
  {
    LogError("ScalersRecover") << 
      "Could not find Level1TriggerScalersCollection";
    return;
  }

  if ( data->size() < 1 )
  {
    LogError("ScalersRecover") << 
      "Could not find Level1TriggerScalers element from Collection";
    return;
  }

  Level1TriggerScalersCollection::const_iterator triggerScalers = 
    data->begin();


  int lumiSection = triggerScalers->lumiSegmentNrLumiSeg();
  if ( ( ( lastLumiSection==-1 ) && ( lumiSection == 1 )) || 
       ( ( lastLumiSection>0 )   && ( lastLumiSection != lumiSection )))
  {
    timespec zeit = triggerScalers->collectionTimeLumiSeg();
    time_t seconds = zeit.tv_sec;
    long int nanos = zeit.tv_nsec;

    hora = gmtime(&seconds);
    strftime(heure,sizeof(heure),"%Y.%m.%d %H:%M:%S", hora);
    sprintf(sNanos,"%9.9d", (int)nanos);

    std::ostringstream insert;
    insert <<  
      "INSERT INTO LEVEL1_TRIGGER_CONDITIONS (RUNNUMBER,LUMISEGMENTNR,TIME,TIME_NS,TIME_STAMP" <<
      ",TRIGGERSPHYSICSGENERATEDFDL" << 
      ",TRIGGERSPHYSICSLOST" << 
      ",TRIGGERSPHYSICSLOSTBEAMACTIVE" << 
      ",TRIGGERSPHYSICSLOSTBEAMINACTI" << 
      ",L1ASPHYSICS" << 
      ",L1ASRANDOM" << 
      ",L1ASTEST" << 
      ",L1ASCALIBRATION" << 
      ",DEADTIME" << 
      ",DEADTIMEBEAMACTIVE" << 
      ",DEADTIMEBEAMACTIVETRIGGERRULE" << 
      ",DEADTIMEBEAMACTIVECALIBRATION" << 
      ",DEADTIMEBEAMACTIVEPRIVATEORBI" << 
      ",DEADTIMEBEAMACTIVEPARTITIONCO" << 
      ",DEADTIMEBEAMACTIVETIMESLOT" << 
      ") VALUES (" << iEvent.run() << 
      "," << lumiSection << 
      "," << zeit.tv_sec << 
      "," << nanos << 
      ",TO_TIMESTAMP('" << heure << "." << sNanos <<
      "','YYYY.MM.DD HH24:MI:SS.FF')" << 
      "," << triggerScalers->triggersPhysicsGeneratedFDL() <<
      "," << triggerScalers->triggersPhysicsLost() << 
      "," << triggerScalers->triggersPhysicsLostBeamActive() << 
      "," << triggerScalers->triggersPhysicsLostBeamInactive() << 
      "," << triggerScalers->l1AsPhysics() << 
      "," << triggerScalers->l1AsRandom() << 
      "," << triggerScalers->l1AsTest() << 
      "," << triggerScalers->l1AsCalibration() << 
      "," << triggerScalers->deadtime() << 
      "," << triggerScalers->deadtimeBeamActive() << 
      "," << triggerScalers->deadtimeBeamActiveTriggerRules() << 
      "," << triggerScalers->deadtimeBeamActiveCalibration() << 
      "," << triggerScalers->deadtimeBeamActivePrivateOrbit() << 
      "," << triggerScalers->deadtimeBeamActivePartitionController() << 
      "," << triggerScalers->deadtimeBeamActiveTimeSlot() << 
      ");" ;
    
    if ( sql != NULL ) { fprintf(sql,"%s\n", insert.str().c_str());}
    
    std::vector<unsigned int> algo = triggerScalers->gtAlgoCounts();
    int length = algo.size();
    for ( int i=0; i<length; i++)
    {
      std::ostringstream ainsert;
      ainsert << "INSERT INTO LEVEL1_TRIGGER_ALGO_CONDITIONS (RUNNUMBER,BIT,LUMISEGMENTNR,TIME,TIME_NS,TIME_STAMP,GTALGOCOUNTS) VALUES (" <<
	iEvent.run() << 
	"," << i << 
	"," << lumiSection << 
	"," << zeit.tv_sec << 
	"," << nanos << 
	",TO_TIMESTAMP('" << heure << "." <<  sNanos << 
	"','YYYY.MM.DD HH24:MI:SS.FF')," 
	      << algo[i] << ");";
      
      if ( sql != NULL ) { fprintf(sql,"%s\n", ainsert.str().c_str());}
    }

    std::vector<unsigned int> tech = triggerScalers->gtAlgoCounts();
    length = tech.size();
    for ( int i=0; i<length; i++)
    {
      std::ostringstream tinsert;
      tinsert << "INSERT INTO LEVEL1_TRIGGER_TECH_CONDITIONS (RUNNUMBER,BIT,LUMISEGMENTNR,TIME,TIME_NS,TIME_STAMP,GTTECHCOUNTS) VALUES (" <<
	    iEvent.run() << 
	    "," << i << 
	    "," << lumiSection << 
	    "," << zeit.tv_sec << 
	    "," << nanos << 
	    ",TO_TIMESTAMP('" << heure << "." <<  sNanos << 
	    "','YYYY.MM.DD HH24:MI:SS.FF')," 
	    << tech[i] << ");";

      if ( sql != NULL ) { fprintf(sql,"%s\n", tinsert.str().c_str());}
    }
    lastLumiSection = lumiSection;
  }
}


void ScalersRecover::beginJob()
{
}

void ScalersRecover::endJob() 
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(ScalersRecover);
