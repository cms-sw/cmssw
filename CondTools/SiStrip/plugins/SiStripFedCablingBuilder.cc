#include "CondTools/SiStrip/plugins/SiStripFedCablingBuilder.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <iostream>
#include <fstream>
#include <sstream>

SiStripFedCablingBuilder::SiStripFedCablingBuilder( const edm::ParameterSet& iConfig){}

void SiStripFedCablingBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup){

  unsigned int run=evt.id().run();

  edm::LogInfo("SiStripFedCablingBuilder") << "... creating dummy SiStripFedCabling Data for Run " << run << "\n " << std::endl;

   edm::ESHandle<SiStripFedCabling> _siStripFedCabling;
   iSetup.get<SiStripFedCablingRcd>().get( _siStripFedCabling ); 

   SiStripFedCabling *obj= new SiStripFedCabling(*(_siStripFedCabling.product()));

  {
    std::stringstream ss;
    ss << "[testSiStripFedCabling::" << __func__ << "]"
       << " VERBOSE DEBUG" << std::endl;
    obj->print( ss );
    ss << std::endl;
    edm::LogVerbatim("testSiStripFedCabling") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[testSiStripFedCabling::" << __func__ << "]"
       << " TERSE DEBUG" << std::endl;
    obj->terse( ss );
    ss << std::endl;
    edm::LogVerbatim("testSiStripFedCabling") << ss.str();
  }
  
  {
    std::stringstream ss;
    ss << "[testSiStripFedCabling::" << __func__ << "]"
       << " SUMMARY DEBUG" << std::endl;
    obj->summary( ss );
    ss << std::endl;
    edm::LogVerbatim("testSiStripFedCabling") << ss.str();
  }
  


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
     
