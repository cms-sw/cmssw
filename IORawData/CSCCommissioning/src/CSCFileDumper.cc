#include <iomanip>

#include "CSCFileDumper.h"

//Framework stuff
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"

//FEDRawData
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include <strstream>

CSCFileDumper::CSCFileDumper(const edm::ParameterSet & pset){
	output = pset.getUntrackedParameter<std::string>("output");
	std::string fedIDs = pset.getUntrackedParameter<std::string>("fedIDs");

	if( fedIDs=="DAQ" || fedIDs=="" ){
		fedID_first = FEDNumbering::getCSCFEDIds().first;
		fedID_last  = FEDNumbering::getCSCFEDIds().second;
	} else
	if( fedIDs=="TF" ){
		fedID_first = FEDNumbering::getCSCTFFEDIds().first;
		fedID_last  = FEDNumbering::getCSCTFFEDIds().second;
	} else
		throw std::runtime_error(std::string("Set CSCFileDumper::fedIDs to either DAQ or TF"));

	//std::string cur_pos = type.rfind(","), prev_pos = std::string::npos;
}

CSCFileDumper::~CSCFileDumper(void){
	std::map<int,FILE*>::const_iterator stream = dump_files.begin();
	while( stream != dump_files.end() ){
		fclose(stream->second);
		stream++;
	}
}

void CSCFileDumper::analyze(const edm::Event & e, const edm::EventSetup& c){
	// Get a handle to the FED data collection
	edm::Handle<FEDRawDataCollection> rawdata;

	// Get a handle to the FED data collection
	e.getByType(rawdata);

	for(int id=fedID_first; id<=fedID_last; ++id){ //for each of our DCCs
		std::map<int,FILE*>::const_iterator stream = dump_files.find(id);
		if( stream == dump_files.end() ){
			std::strstream name;
			name<<output<<"_"<<id<<std::ends;
			FILE *file;
			if( (file = fopen(name.str(),"wt"))==NULL ){
				std::cout<<"Cannot open the file: "<<name.str()<<std::endl;
				continue;
			} else
				dump_files[id] = file;
			stream = dump_files.find(id);
		}

		/// Take a reference to this FED's data
		const FEDRawData& fedData = rawdata->FEDData(id);
		unsigned short int length = fedData.size();

		if(length){
			//std::cout<<"Event found for fed id:"<<id<<std::endl;
			// Event buffer
			size_t size=length/2;
			const unsigned short *buf = (unsigned short *)fedData.data();
			fwrite(buf,2,size,stream->second);
		}
	}
}
