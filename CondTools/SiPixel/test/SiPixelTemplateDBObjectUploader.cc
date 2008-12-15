#include "CondTools/SiPixel/test/SiPixelTemplateDBObjectUploader.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

SiPixelTemplateDBObjectUploader::SiPixelTemplateDBObjectUploader(const edm::ParameterSet& iConfig):
	theFileNums( iConfig.getParameter<vstring>("fileNums") ),
	theVersion( iConfig.getParameter<double>("Version") )
{
}


SiPixelTemplateDBObjectUploader::~SiPixelTemplateDBObjectUploader()
{
}

void 
SiPixelTemplateDBObjectUploader::beginJob(const edm::EventSetup&)
{
}

void
SiPixelTemplateDBObjectUploader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

void 
SiPixelTemplateDBObjectUploader::endJob()
{
	//--- Make the POOL-ORA object to store the database object
	SiPixelTemplateDBObject* obj = new SiPixelTemplateDBObject;
  obj->fillDB(theFileNums);
	obj->setVersion(theVersion);
	
	// Uncomment to output the contents of the db object at the end of the job
	//std::cout << *obj << std::endl;

  //--- Create a new IOV
	edm::Service<cond::service::PoolDBOutputService> poolDbService;
	
  if( poolDbService.isAvailable() ) {
    if ( poolDbService->isNewTagRequest("SiPixelTemplateDBObjectRcd") )
      poolDbService->
				createNewIOV<SiPixelTemplateDBObject>( obj,
					poolDbService->beginOfTime(),
					poolDbService->endOfTime(),
					"SiPixelTemplateDBObjectRcd"  );
    else
      poolDbService->
				appendSinceTime<SiPixelTemplateDBObject>( obj, 
					poolDbService->currentTime(),
					"SiPixelTemplateDBObjectRcd" );
  }
  else {
    std::cout << "Pool Service Unavailable" << std::endl;
    // &&& throw an exception???
	}
}

