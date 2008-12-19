#include "XMLGeometryBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/GeometryObjects/interface/GeometryFile.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"


#include <iostream>
#include <string>
#include <vector>
#include <fstream>

XMLGeometryBuilder::XMLGeometryBuilder(const edm::ParameterSet& iConfig)
{
  fname = iConfig.getUntrackedParameter<std::string>("XMLFileName","test.xml");
  zip = iConfig.getUntrackedParameter<bool>("ZIP",true);
}

XMLGeometryBuilder::~XMLGeometryBuilder()
{

}

void
XMLGeometryBuilder::beginJob( edm::EventSetup const& es) 
{
  std::cout<<"XMLGeometryBuilder::beginJob"<<std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    edm::LogError("XMLGeometryBuilder")<<"PoolDBOutputService unavailable";
    return;
  }

  GeometryFile* pgf= new GeometryFile(fname,zip);

  if ( mydbservice->isNewTagRequest("GeometryFileRcd") ) {
    mydbservice->createNewIOV<GeometryFile>( pgf,mydbservice->beginOfTime(),mydbservice->endOfTime(),"GeometryFileRcd");
  } else {
    edm::LogError("XMLGeometryBuilder")<<"GeometryFileRcs Tag already exist";
  }
}
  
