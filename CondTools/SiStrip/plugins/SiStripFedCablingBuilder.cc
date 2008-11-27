#include "CondTools/SiStrip/plugins/SiStripFedCablingBuilder.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <iostream>
#include <fstream>
#include <sstream>

// -----------------------------------------------------------------------------
// 
SiStripFedCablingBuilder::SiStripFedCablingBuilder( const edm::ParameterSet& pset ) :
  printFecCabling_( pset.getUntrackedParameter<bool>("PrintFecCabling",false) ),
  printDetCabling_( pset.getUntrackedParameter<bool>("PrintDetCabling",false) )
{;}

// -----------------------------------------------------------------------------
// 
void SiStripFedCablingReader::beginRun( const edm::Run& run, 
					const edm::EventSetup& setup ) {

  edm::LogInfo("SiStripFedCablingBuilder") << "... creating dummy SiStripFedCabling Data for Run " << run.run() << "\n " << std::endl;

  edm::ESHandle<SiStripFedCabling> _siStripFedCabling;
  iSetup.get<SiStripFedCablingRcd>().get( _siStripFedCabling ); 
   
  edm::LogVerbatim("SiStripFedCablingBuilder") 
    << "[SiStripFedCablingBuilder::" << __func__ << "]"
    << " Building FED cabling...";
  SiStripFedCabling* fed = new SiStripFedCabling( *( _siStripFedCabling.product() ) );
  edm::LogVerbatim("SiStripFedCablingBuilder") 
    << "[SiStripFedCablingBuilder::" << __func__ << "]"
    << " Building FEC cabling...";
  SiStripFecCabling* fec = new SiStripFecCabling( *fed );
  edm::LogVerbatim("SiStripFedCablingBuilder") 
    << "[SiStripFedCablingBuilder::" << __func__ << "]"
    << " Building DET cabling...";
  SiStripDetCabling* det = new SiStripDetCabling( *fed );
  edm::LogVerbatim("SiStripFedCablingBuilder") 
    << "[SiStripFedCablingBuilder::" << __func__ << "]"
    << " Finished building FED/FEC/DET cabling...";
   
  if ( !fed || !fec || !det ) {
    edm::LogError("SiStripFedCablingBuilder") 
      << " NULL pointer for at least one of"
      << " FED/FEC/DET cabling objects: "
      << fed << "/" << fec << "/" << det;
  }
  
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingBuilder::" << __func__ << "]"
       << " VERBOSE DEBUG" << std::endl;
    fed->print( ss );
    ss << std::endl;
    if ( printFecCabling_ ) { fec->print( ss ); }
    ss << std::endl;
    if ( printDetCabling_ ) { det->print( ss ); }
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingBuilder") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingBuilder::" << __func__ << "]"
       << " TERSE DEBUG" << std::endl;
    fed->terse( ss );
    ss << std::endl;
    if ( printFecCabling_ ) { fec->terse( ss ); }
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingBuilder") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[SiStripFedCablingBuilder::" << __func__ << "]"
       << " SUMMARY DEBUG" << std::endl;
    fed->summary( ss );
    ss << std::endl;
    edm::LogVerbatim("SiStripFedCablingBuilder") << ss.str();
  }
  


  //End now write sistripnoises data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if( mydbservice.isAvailable() ){
    if ( mydbservice->isNewTagRequest("SiStripFedCablingRcd") ){
      mydbservice->createNewIOV<SiStripFedCabling>(fed,mydbservice->beginOfTime(),mydbservice->endOfTime(),"SiStripFedCablingRcd");
    } else {  
      mydbservice->appendSinceTime<SiStripFedCabling>(fed,mydbservice->currentTime(),"SiStripFedCablingRcd");      
    }
  }else{
    edm::LogError("SiStripFedCablingBuilder")<<"Service is unavailable"<<std::endl;
  }
}
     
