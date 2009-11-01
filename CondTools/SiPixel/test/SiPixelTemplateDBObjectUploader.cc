#include "CondTools/SiPixel/test/SiPixelTemplateDBObjectUploader.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <fstream>

SiPixelTemplateDBObjectUploader::SiPixelTemplateDBObjectUploader(const edm::ParameterSet& iConfig):
	theTemplateCalibrations( iConfig.getParameter<vstring>("siPixelTemplateCalibrations") ),
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

	// Local variables 
	const char *tempfile;
	int m;
	
	// Set the number of templates to be passed to the dbobject
	obj->setNumOfTempl(theTemplateCalibrations.size());

	// Set the version of the template dbobject - this is an external parameter
	obj->setVersion(theVersion);

	//  open the template file(s) 
	for(m=0; m< obj->numOfTempl(); ++m){

		edm::FileInPath file( theTemplateCalibrations[m].c_str() );
		tempfile = (file.fullPath()).c_str();

		std::ifstream in_file(tempfile, std::ios::in);
			
		if(in_file.is_open()){
			edm::LogInfo("SiPixelTemplateDBObjectUploader") << "Opened Template File: " << file.fullPath().c_str() << std::endl;

			// Local variables 
			char title_char[80], c;
			SiPixelTemplateDBObject::char2float temp;
			float tempstore;
			int iter,j;
			
			// Templates contain a header char - we must be clever about storing this
			for (iter = 0; (c=in_file.get()) != '\n'; ++iter) {
				if(iter < 79) {title_char[iter] = c;}
			}
			if(iter > 78) {iter=78;}
			title_char[iter+1] ='\n';
			
			for(j=0; j<80; j+=4) {
				temp.c[0] = title_char[j];
				temp.c[1] = title_char[j+1];
				temp.c[2] = title_char[j+2];
				temp.c[3] = title_char[j+3];
				obj->push_back(temp.f);
				obj->setMaxIndex(obj->maxIndex()+1);
			}
			
			// Fill the dbobject
			in_file >> tempstore;
			while(!in_file.eof()) {
				obj->setMaxIndex(obj->maxIndex()+1);
				obj->push_back(tempstore);
				in_file >> tempstore;
			}
			

			in_file.close();
		}
		else {
			// If file didn't open, report this
			edm::LogError("SiPixelTemplateDBObjectUploader") << "Error opening File" << tempfile << std::endl;
		}
	}
		
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

