#include "XMLMagneticFieldGeometryBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <string>
#include <vector>
#include <fstream>

XMLMagneticFieldGeometryBuilder::XMLMagneticFieldGeometryBuilder( const edm::ParameterSet& iConfig )
{
  fname = iConfig.getUntrackedParameter<std::string>("XMLFileName","test.xml");
  zip = iConfig.getUntrackedParameter<bool>("ZIP",true);
}

XMLMagneticFieldGeometryBuilder::~XMLMagneticFieldGeometryBuilder()
{
}

void
XMLMagneticFieldGeometryBuilder::beginJob() 
{
  edm::LogInfo( "XMLMagneticFieldGeometryBuilder" ) << "XMLMagneticFieldGeometryBuilder::beginJob";
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if( !mydbservice.isAvailable())
  {
    edm::LogError( "XMLMagneticFieldGeometryBuilder" ) << "PoolDBOutputService unavailable";
    return;
  }

  FileBlob* pgf= new FileBlob( fname, zip );

  if( mydbservice->isNewTagRequest( "IdealMagneticFieldRecord"))
  {
    mydbservice->createNewIOV<FileBlob>( pgf, mydbservice->beginOfTime(), mydbservice->endOfTime(), "IdealMagneticFieldRecord" );
  }
  else
  {
    edm::LogError( "XMLMagneticFieldGeometryBuilder" ) << "GeometryFileRcd Tag already exist";
  }
}
  
