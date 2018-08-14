#include "CondTools/SiStrip/plugins/SiStripFedCablingBuilder.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripFecCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Run.h"
#include <iostream>
#include <fstream>
#include <sstream>

// -----------------------------------------------------------------------------
// 
SiStripFedCablingBuilder::SiStripFedCablingBuilder( const edm::ParameterSet& pset ) :
  printFecCabling_( pset.getUntrackedParameter<bool>("PrintFecCabling",false) ),
  printDetCabling_( pset.getUntrackedParameter<bool>("PrintDetCabling",false) ),
  printRegionCabling_( pset.getUntrackedParameter<bool>("PrintRegionCabling",false) )
{;}

// -----------------------------------------------------------------------------
// 
void SiStripFedCablingBuilder::beginRun( const edm::Run& run, 
					 const edm::EventSetup& setup ) {

  edm::LogInfo("SiStripFedCablingBuilder") << "... creating dummy SiStripFedCabling Data for Run " << run.run() << "\n " << std::endl;

  edm::LogVerbatim("SiStripFedCablingBuilder") 
    << "[SiStripFedCablingBuilder::" << __func__ << "]"
    << " Retrieving FED cabling...";
  edm::ESHandle<SiStripFedCabling> fed;
  setup.get<SiStripFedCablingRcd>().get( fed ); 

  edm::LogVerbatim("SiStripFedCablingBuilder") 
    << "[SiStripFedCablingBuilder::" << __func__ << "]"
    << " Retrieving FEC cabling...";
  edm::ESHandle<SiStripFecCabling> fec;
  setup.get<SiStripFecCablingRcd>().get( fec ); 

  edm::LogVerbatim("SiStripFedCablingBuilder") 
    << "[SiStripFedCablingBuilder::" << __func__ << "]"
    << " Retrieving DET cabling...";
  edm::ESHandle<SiStripDetCabling> det;
  setup.get<SiStripDetCablingRcd>().get( det ); 

  edm::LogVerbatim("SiStripFedCablingBuilder") 
    << "[SiStripFedCablingBuilder::" << __func__ << "]"
    << " Retrieving REGION cabling...";
  edm::ESHandle<SiStripRegionCabling> region;
  setup.get<SiStripRegionCablingRcd>().get( region ); 

  if ( !fed.isValid() ) {
    edm::LogError("SiStripFedCablingBuilder") 
      << " Invalid handle to FED cabling object: ";
    return;
  }
  
  edm::ESHandle<TrackerTopology> tTopo;
  setup.get<TrackerTopologyRcd>().get(tTopo);
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingBuilder::" << __func__ << "]"
       << " VERBOSE DEBUG" << std::endl;
    fed->print(ss, tTopo.product());
    ss << std::endl;
    if ( printFecCabling_ && fec.isValid() ) { fec->print( ss ); }
    ss << std::endl;
    if ( printDetCabling_ && det.isValid() ) { det->print( ss ); }
    ss << std::endl;
    if ( printRegionCabling_ && region.isValid() ) { region->print( ss ); }
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingBuilder") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingBuilder::" << __func__ << "]"
       << " TERSE DEBUG" << std::endl;
    fed->terse( ss );
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingBuilder") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingBuilder::" << __func__ << "]"
       << " SUMMARY DEBUG" << std::endl;
    fed->summary(ss, tTopo.product());
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingBuilder") << ss.str();
  }
  
  edm::LogVerbatim("SiStripFedCablingBuilder") 
    << "[SiStripFedCablingBuilder::" << __func__ << "]"
    << " Copying FED cabling...";
  SiStripFedCabling* obj = new SiStripFedCabling( *( fed.product() ) );
  
  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if( mydbservice.isAvailable() ){
    if ( mydbservice->isNewTagRequest("SiStripFedCablingRcd") ){
      mydbservice->createNewIOV<SiStripFedCabling>(obj,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripFedCablingRcd");
    } else {  
      mydbservice->appendSinceTime<SiStripFedCabling>(obj,mydbservice->currentTime(),"SiStripFedCablingRcd");      
    }
  }else{
    edm::LogError("SiStripFedCablingBuilder")<<"Service is unavailable"<<std::endl;
  }
}
     
