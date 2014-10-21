#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeGenErrorDBObjectESSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <fstream>

SiPixelFakeGenErrorDBObjectESSource::SiPixelFakeGenErrorDBObjectESSource(const edm::ParameterSet& conf_) : GenErrorCalibrations_(conf_.getParameter<vstring>("siPixelGenErrorCalibrations")),
																																																					 version_(conf_.getParameter<double>("Version"))
{
	edm::LogInfo("SiPixelFakeGenErrorDBObjectESSource::SiPixelFakeGenErrorDBObjectESSource");
	//the following line is needed to tell the framework what
	// data is being produced
	setWhatProduced(this);
	findingRecord<SiPixelGenErrorDBObjectRcd>();
}

SiPixelFakeGenErrorDBObjectESSource::~SiPixelFakeGenErrorDBObjectESSource()
{
}

std::auto_ptr<SiPixelGenErrorDBObject> SiPixelFakeGenErrorDBObjectESSource::produce(const SiPixelGenErrorDBObjectRcd & )
{
	using namespace edm::es;

	//Mostly copied from CondTools/SiPixel/test/SiPixelGenErrorDBObjectUploader.cc
	//--- Make the POOL-ORA object to store the database object
	SiPixelGenErrorDBObject* obj = new SiPixelGenErrorDBObject;

	// Local variables 
	const char *tempfile;
	int m;
	
	// Set the number of GenErrors to be passed to the dbobject
	obj->setNumOfTempl(GenErrorCalibrations_.size());

	// Set the version of the GenError dbobject - this is an external parameter
	obj->setVersion(version_);

	//  open the GenError file(s) 
	for(m=0; m< obj->numOfTempl(); ++m){

		edm::FileInPath file( GenErrorCalibrations_[m].c_str() );
		tempfile = (file.fullPath()).c_str();

		std::ifstream in_file(tempfile, std::ios::in);
			
		if(in_file.is_open()){
			edm::LogInfo("SiPixelFakeGenErrorDBObjectESSource") << "Opened GenError File: " << file.fullPath().c_str() << std::endl;

			// Local variables 
			char title_char[80], c;
			SiPixelGenErrorDBObject::char2float temp;
			float tempstore;
			int iter,j;
			
			// GenErrors contain a header char - we must be clever about storing this
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
			edm::LogError("SiPixeFakelGenErrorDBObjectESSource") << "Error opening File" << tempfile << std::endl;
		}
	}

	//std::cout << *obj << std::endl;
	return std::auto_ptr<SiPixelGenErrorDBObject>(obj);
}

void SiPixelFakeGenErrorDBObjectESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
	const edm::IOVSyncValue& iosv, 
	edm::ValidityInterval& oValidity ) {
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;  
}

