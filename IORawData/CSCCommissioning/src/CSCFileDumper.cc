#include <iomanip>

#include "CSCFileDumper.h"

//Framework stuff
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//FEDRawData
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <sstream>
#include <stdio.h>

CSCFileDumper::CSCFileDumper(const edm::ParameterSet & pset)
{
	output = pset.getUntrackedParameter<std::string>("output");
	fedID_first = FEDNumbering::MINCSCFEDID;
	fedID_last  = FEDNumbering::MAXCSCTFFEDID;
	events = pset.getUntrackedParameter<std::string>("events","");
	source_ = pset.getUntrackedParameter<std::string>("source","rawDataCollector");

	if( events.length() ){
		for(size_t pos1=0,pos2=events.find(",");;pos1=pos2+1,pos2=events.find(",",pos2+1)){
			if( pos2!=std::string::npos ){
				long event=0;
				if( sscanf(events.substr(pos1,pos2-pos1).c_str(),"%ld",&event) == 1 && event>=0 )
					eventsToDump.insert((unsigned long)event);
				else
					edm::LogError("CSCFileDumper")<<" cannot parse events ("<<events<<") parameter: "<<events.substr(pos1,pos2-pos1);
			} else {
				long event=0;
				if( sscanf(events.substr(pos1,events.length()-pos1).c_str(),"%ld",&event) == 1 && event>=0 )
					eventsToDump.insert((unsigned long)event);
				else
					edm::LogError("CSCFileDumper")<<" cannot parse events ("<<events<<") parameter: "<<events.substr(pos1,events.length()-pos1);
				break;
			}
		}
		std::ostringstream tmp;
		for(std::set<unsigned long>::const_iterator evt=eventsToDump.begin(); evt!=eventsToDump.end(); evt++) tmp<<*evt<<" ";
		edm::LogInfo("CSCFileDumper")<<" Following events will be dumped: "<<tmp.str();
	} else
		edm::LogInfo("CSCFileDumper")<<" All events will be dumped";
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
	e.getByLabel(source_, rawdata);

	for(int id=fedID_first; id<=fedID_last; ++id){ //for each of our DCCs
		std::map<int,FILE*>::const_iterator stream = dump_files.find(id);
		if( stream == dump_files.end() ){
			std::ostringstream name;
			name<<output<<"_"<<id<<std::ends;
			FILE *file;
			if( (file = fopen(name.str().c_str(),"wt"))==NULL ){
				edm::LogError("CSCFileDumper")<<"Cannot open the file: "<<name.str();
				continue;
			} else
				dump_files[id] = file;
			stream = dump_files.find(id);
		}

		/// Take a reference to this FED's data
		const FEDRawData& fedData = rawdata->FEDData(id);
		unsigned short int length = fedData.size();

		if( length && ( eventsToDump.empty() || eventsToDump.find( (unsigned long)e.id().event() )!=eventsToDump.end() ) ){
			//std::cout<<"Event found for fed id:"<<id<<std::endl;
			// Event buffer
			size_t size=length/2;
			const unsigned short *buf = (unsigned short *)fedData.data();
			fwrite(buf,2,size,stream->second);
		}
	}
}
