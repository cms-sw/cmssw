#include "XMLGeometryBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"


#include <iostream>
#include <string>
#include <vector>
#include <fstream>

XMLGeometryBuilder::XMLGeometryBuilder(const edm::ParameterSet& iConfig)
{
  fname = iConfig.getUntrackedParameter<std::string>("XMLFileName","test.xml");
  zip = iConfig.getUntrackedParameter<bool>("ZIP",true);
  record = iConfig.getUntrackedParameter<std::string>("record","GeometryFileRcd");
}

XMLGeometryBuilder::~XMLGeometryBuilder()
{

}

void
XMLGeometryBuilder::beginJob() 
{
  std::cout<<"XMLGeometryBuilder::beginJob"<<std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable() ){
    edm::LogError("XMLGeometryBuilder")<<"PoolDBOutputService unavailable";
    return;
  }

  FileBlob* pgf= new FileBlob(fname,zip);

  if ( mydbservice->isNewTagRequest(record) ) {
    mydbservice->createNewIOV<FileBlob>( pgf,mydbservice->beginOfTime(),mydbservice->endOfTime(),record);
  } else {
    edm::LogError("XMLGeometryBuilder")<<"GeometryFileRcd Tag already exist";
  }
}
  
