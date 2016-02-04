// -*- C++ -*-
//
// Package:    SiPixelQualityESProducer
// Class:      SiPixelQualityESProducer
// 
/**\class SiPixelQualityESProducer SiPixelQualityESProducer.h CalibTracker/SiPixelESProducer/src/SiPixelQualityESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author: Gemma Tinti
//         Created:  Jan 13 2011
//
//

// user include files

#include <cassert>
#include "CalibTracker/SiPixelESProducers/interface/SiPixelQualityESProducer.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityFromDbRcd.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
//
// constructors and destructor
//
using namespace edm;

SiPixelQualityESProducer::SiPixelQualityESProducer(const edm::ParameterSet& conf_) 
  : //fp_(conf_.getParameter<edm::FileInPath>("file")),
    toGet(conf_.getParameter<Parameters>("ListOfRecordToMerge"))
{
   edm::LogInfo("SiPixelQualityESProducer::SiPixelQualityESProducer");
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  findingRecord<SiPixelQualityRcd>();
}


SiPixelQualityESProducer::~SiPixelQualityESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

std::auto_ptr<SiPixelQuality> SiPixelQualityESProducer::produce(const SiPixelQualityRcd & iRecord)
{
  
  std::string recordName;

  ///////////////////////////////////////////////////////
  //  errortype "whole" = int 0 in DB  BadRocs = 65535 //
  //  errortype "tbmA" = int 1 in DB  BadRocs = 255    //
  //  errortype "tbmB" = int 2 in DB  Bad Rocs = 65280 //
  //  errortype "none" = int 3 in DB                   //
  ///////////////////////////////////////////////////////
  
  //if I have understood this is the one got from the DB or file, but in any case the ORIGINAL(maybe i need to get the record for it)
  //SiPixelQuality * obj = new SiPixelQuality();
  //SiPixelQuality::disabledModuleType BadModule;
  //BadModule.DetID = 1; BadModule.errorType = 0; BadModule.BadRocs = 65535; obj->addDisabledModule(BadModule);
  
    //start with the record thta existed already to decouple the debugging
  //here i can do whatever i need with the detVoff
 
  edm::ESHandle<SiStripDetVOff> Voff;
  edm::ESHandle<SiPixelQuality> dbobject;
  
  for( Parameters::iterator itToGet = toGet.begin(); itToGet != toGet.end(); ++itToGet ) {

    recordName = itToGet->getParameter<std::string>("record");
    
    if (recordName=="SiPixelDetVOffRcd")
      iRecord.getRecord<SiPixelDetVOffRcd>().get(Voff); 
    if (recordName=="SiPixelQualityFromDbRcd")
      iRecord.getRecord<SiPixelQualityFromDbRcd>().get(dbobject);
  } //end getting the records from the parameters
  
  //now the dbobject is the one copied from the db
  //here make a copy of dbobject, but now the label has to be empty not to interfeare with the Reco
  std::auto_ptr<SiPixelQuality> dbptr(new SiPixelQuality(*(dbobject)));
  
  //here is the magic line in which it switches off Bad Modules
  dbptr->add(Voff.product());

  return dbptr;
}

void SiPixelQualityESProducer::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;  
}
