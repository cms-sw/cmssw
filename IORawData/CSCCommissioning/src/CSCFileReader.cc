#include "CSCFileReader.h"

//#include <iostream.h>

#include <errno.h>
#include <stdlib.h>
#include <cstring>

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/Provenance/interface/EventID.h>
#include <DataFormats/Provenance/interface/Timestamp.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <string>
#include <iosfwd>
#include <sstream>
#include <iostream>
#include <algorithm>

#define nRUIs 40
#define nFUs  4

CSCFileReader::CSCFileReader(const edm::ParameterSet& pset):DaqBaseReader(){
	LogDebug("CSCFileReader|ctor")<<"Started ...";
	// Below some data members are recycled for both cases: RUIs and FUs
	//  this is ok as long as eighter of RUI or FU are are provided in .cfg (not both)
	nActiveRUIs = 0;
	nActiveFUs  = 0;
	for(int unit=0; unit<nRUIs; unit++){
		std::ostringstream ruiName, fuName;
		ruiName<<"RUI"<<(unit<10?"0":"")<<unit<<std::ends;
		fuName <<"FU" <<unit<<std::ends;
		std::vector<std::string> ruiFiles = pset.getUntrackedParameter< std::vector<std::string> >(ruiName.str().c_str(),std::vector<std::string>(0));
		std::vector<std::string> fuFiles  = pset.getUntrackedParameter< std::vector<std::string> >(fuName.str().c_str(),std::vector<std::string>(0));
		if( ruiFiles.begin() != ruiFiles.end() ) nActiveRUIs++;
		if( fuFiles.begin()  != fuFiles.end()  ) nActiveFUs++;
	}
	if( nActiveFUs && nActiveRUIs )
		throw cms::Exception("CSCFileReader|configuration")<<"RUIs and FUs in conflict: either RUI or FU may be defined at a time, not both";
	if( !nActiveFUs && !nActiveRUIs )
		throw cms::Exception("CSCFileReader|configuration")<<"Module lacks configuration";


	// Get list of RUI input files from .cfg file
	for(int rui=0; rui<nRUIs && !nActiveFUs; rui++){
		std::ostringstream name;
		name<<"RUI"<<(rui<10?"0":"")<<rui<<std::ends;

		// Obtain list of files associated with current RUI
		fileNames[rui] = pset.getUntrackedParameter< std::vector<std::string> >(name.str().c_str(),std::vector<std::string>(0));
		currentFile[rui] = fileNames[rui].begin();

		// If list of files is not empty, open first file
		if( currentFile[rui] != fileNames[rui].end() ){
			try {
				RUI[rui].open(currentFile[rui]->c_str());
			} catch ( std::runtime_error err ){
				throw cms::Exception("CSCFileReader")<<"InputFileMissing: "<<err.what()<<" (errno="<<errno<<")";
			}
			nActiveRUIs++;
		}

		// Filter out possible corruptions
		RUI[rui].reject(FileReaderDDU::DDUoversize|FileReaderDDU::FFFF|FileReaderDDU::Unknown);
		// Do not select anything in particular
		RUI[rui].select(0);

		currentL1A[rui] = -1;
	}

	// Get list of FU input files from .cfg file
	for(int fu=0; fu<nFUs && !nActiveRUIs; fu++){
		std::ostringstream name;
		name<<"FU"<<fu<<std::ends;

		// Obtain list of files associated with current FU
		fileNames[fu] = pset.getUntrackedParameter< std::vector<std::string> >(name.str().c_str(),std::vector<std::string>(0));
		currentFile[fu] = fileNames[fu].begin();

		// If list of files is not empty, open first file
		if( currentFile[fu] != fileNames[fu].end() ){
			try {
				FU[fu].open(currentFile[fu]->c_str());
			} catch ( std::runtime_error err ){
				throw cms::Exception("CSCFileReader")<<"InputFileMissing: "<<err.what()<<" (errno="<<errno<<")";
			}
			nActiveFUs++;
		}

		// Filter out possible corruptions
		FU[fu].reject(FileReaderDCC::DCCoversize|FileReaderDCC::FFFF|FileReaderDCC::Unknown);
		// Do not select anything in particular
		FU[fu].select(0);

		currentL1A[fu] = -1;
	}

	if( nActiveRUIs && !nActiveFUs ){
		// Assign RUIs to FEDs
		for(int fed=FEDNumbering::MINCSCFEDID; fed<=FEDNumbering::MAXCSCFEDID; fed++){
			std::ostringstream name;
			name<<"FED"<<fed<<std::ends;
			std::vector<std::string> rui_list = pset.getUntrackedParameter< std::vector<std::string> >(name.str().c_str(),std::vector<std::string>(0));
			for(std::vector<std::string>::const_iterator rui=rui_list.begin(); rui!=rui_list.end(); rui++)
				FED[fed].push_back((unsigned int)atoi(rui->c_str()+rui->length()-2));
		}
		// Do the same for Track-Finder FED
		for(int fed=FEDNumbering::MINCSCTFFEDID; fed<=FEDNumbering::MAXCSCTFFEDID; fed++){
			std::ostringstream name;
			name<<"FED"<<fed<<std::ends;
			std::vector<std::string> rui_list = pset.getUntrackedParameter< std::vector<std::string> >(name.str().c_str(),std::vector<std::string>(0));
			for(std::vector<std::string>::const_iterator rui=rui_list.begin(); rui!=rui_list.end(); rui++)
				FED[fed].push_back((unsigned int)atoi(rui->c_str()+rui->length()-2));
		}
	}
	// Starting point
	firstEvent = pset.getUntrackedParameter<int>("firstEvent",0);
	nEvents = 0;
	expectedNextL1A = -1;

	// If Track-Finder was in readout specify its position in the record or set -1 otherwise
	//  Current agriment is that if there is a TF event it is first DDU record
	tfDDUnumber = pset.getUntrackedParameter<int>("tfDDUnumber",-1);

	// For efficiency reasons create this big chunk of data only once
	tmpBuf = new unsigned short[200000*nRUIs+4*4];
	// Event buffer and its length for every FU
	fuEvent[0]=0; fuEventSize[0]=0;
	fuEvent[1]=0; fuEventSize[1]=0;
	fuEvent[2]=0; fuEventSize[2]=0;
	fuEvent[3]=0; fuEventSize[3]=0;
	// Event buffer and its length for every RU
	for(int rui=0; rui<nRUIs; rui++){
		ruBuf[rui] = 0;
		ruBufSize[rui] = 0;
	}
	LogDebug("CSCFileReader|ctor")<<"... and finished";
}

CSCFileReader::~CSCFileReader(void){ if(tmpBuf) delete [] tmpBuf; }

int CSCFileReader::readRUI(int rui, const unsigned short* &buf, size_t &length){
	if( currentFile[rui] == fileNames[rui].end() ) return -1;
	do {
		try {
			length = RUI[rui].next(buf);
		} catch ( std::runtime_error err ){
			throw cms::Exception("CSCFileReader|reading")<<"EndOfStream: "<<err.what()<<" (errno="<<errno<<")";
		}
		if( length==0 ){ // end of file, try next one
			if( ++currentFile[rui] != fileNames[rui].end() ){
				try {
					RUI[rui].open(currentFile[rui]->c_str());
				} catch ( std::runtime_error err ){
					throw cms::Exception("CSCFileReader|reading")<<"InputFileMissing: "<<err.what()<<" (errno="<<errno<<")";
				}
			} else return -1;
		}
	} while( length==0 );
	return buf[2]|((buf[3]&0xFF)<<16);
}

int CSCFileReader::readFU(int fu, const unsigned short* &buf, size_t &length){
	if( currentFile[fu] == fileNames[fu].end() ) return -1;
	do {
		try {
			length = FU[fu].next(buf);
		} catch ( std::runtime_error err ){
			throw cms::Exception("CSCFileReader|reading")<<"EndOfStream: "<<err.what()<<" (errno="<<errno<<")";
		}
		if( length==0 ){ // end of file, try next one
			if( ++currentFile[fu] != fileNames[fu].end() ){
				try {
					FU[fu].open(currentFile[fu]->c_str());
				} catch ( std::runtime_error err ){
					throw cms::Exception("CSCFileReader|reading")<<"InputFileMissing: "<<err.what()<<" (errno="<<errno<<")";
				}
			} else return -1;
		}
	} while( length==0 );
	// Take L1A from first DDU header in the DCC record (shift=8)
	return buf[2+8]|((buf[3+8]&0xFF)<<16);
}

int CSCFileReader::buildEventFromRUIs(FEDRawDataCollection *data){
	int eventNumber =-1; // Will determine below

	do {
		// Read next event from RUIs
		for(int rui=0; rui<nRUIs; rui++){
			// read event from the RUI only in two cases:
			//     1) it is readable (currentL1A>0) and we expect next event from the RUI
			//     2) it is first time (expectedNextL1A<0)
			if((currentL1A[rui]>0 && currentL1A[rui]<expectedNextL1A) || expectedNextL1A<0 )
				currentL1A[rui] = readRUI(rui,ruBuf[rui],ruBufSize[rui]);
		}
		eventNumber =-1;

		// Select lowest L1A from all RUIs and don't expect next event from RUIs that currently hold higher L1A
		for(int rui=0; rui<nRUIs; rui++)
			if( currentL1A[rui]>=0 && (eventNumber>currentL1A[rui] || eventNumber==-1) ) eventNumber=currentL1A[rui];
		// No readable RUIs => fall out
		if( eventNumber<0 ) return -1;
		// Expect next event to be incremented by 1 wrt. to the current event
		expectedNextL1A = eventNumber+1;

	} while(nEvents++<firstEvent);

	for(std::map<unsigned int,std::list<unsigned int> >::const_iterator fed=FED.begin(); fed!=FED.end(); fed++)
		if( fed->first<(unsigned int)FEDNumbering::MINCSCTFFEDID ){
			// Now let's pretend that DDU data were wrapped with DCC Header (2 64-bit words) and Trailer (2 64-bit words):
			unsigned short *dccBuf=tmpBuf, *dccCur=dccBuf;
			dccCur[3] = 0x5000; dccCur[2] = 0x0000; dccCur[1] = 0x0000; dccCur[0] = 0x005F; // Fake DCC Header 1
			dccCur[7] = 0xD900; dccCur[6] = 0x0000; dccCur[5] = 0x0000; dccCur[4] = 0x0017; // Fake DCC Header 2
			dccCur += 8;

			for(std::list<unsigned int>::const_iterator rui=fed->second.begin(); rui!=fed->second.end(); rui++){
//cout<<"Event:"<<eventNumber<<"  FED:"<<fed->first<<"  RUI:"<<*(fed->second.begin())<<" currL1A:"<<currentL1A[*rui]<<endl;
				if( currentL1A[*rui]==eventNumber ){
					if(dccCur-dccBuf+ruBufSize[*rui]>=200000*nRUIs+8) throw cms::Exception("CSCFileReader|eventBuffer")<<"OutOfBuffer: Event size exceeds maximal size allowed!";
					memcpy(dccCur,ruBuf[*rui],ruBufSize[*rui]*sizeof(unsigned short));
					dccCur += ruBufSize[*rui];
				}
			}
			dccCur[3] = 0xEF00; dccCur[2] = 0x0000; dccCur[1] = 0x0000; dccCur[0] = 0x0000; // Fake DCC Trailer 2
			dccCur[7] = 0xAF00; dccCur[6] = 0x0000; dccCur[5] = 0x0000; dccCur[4] = 0x0007; // Fake DCC Trailer 2
			dccCur += 8;

			FEDRawData& fedRawData = data->FEDData(fed->first);
			fedRawData.resize((dccCur-dccBuf)*sizeof(unsigned short));
			std::copy((unsigned char*)dccBuf,(unsigned char*)dccCur,fedRawData.data());
		} else {
			for(std::list<unsigned int>::const_iterator rui=fed->second.begin(); rui!=fed->second.end(); rui++){
				FEDRawData& fedRawData = data->FEDData(fed->first);
				fedRawData.resize(ruBufSize[*rui]*sizeof(unsigned short));
				std::copy((unsigned char*)ruBuf[*rui],(unsigned char*)(ruBuf[*rui]+ruBufSize[*rui]),fedRawData.data());
			}
		}

	return eventNumber;
}

int CSCFileReader::nextEventFromFUs(FEDRawDataCollection *data){
	int eventNumber =-1; // Will determine below

	// If this is a first time - read one event from each FU
	if( expectedNextL1A<0 )
		for(int fu=0; fu<nFUs; fu++)
			currentL1A[fu] = readFU(fu,fuEvent[fu],fuEventSize[fu]);

	// Keep buffers for every FU ready at all times
	// When buffer from some FU is ready to go as the next event,
	//  release it, but read next one
	int readyToGo = -1;
	for(int fu=0; fu<nFUs; fu++){
		// If FU is readable and (first loop of this cycle or current FU holds smallest L1A)
		if(currentL1A[fu]>=0 && (eventNumber<0 || currentL1A[fu]<eventNumber)){
			readyToGo   = fu;
			eventNumber = currentL1A[fu];
		}
	}
	// No readable FUs => fall out
	if( readyToGo<0 ) return -1;

	expectedNextL1A = eventNumber + 1;

	// Compose event from DDC record striped of Track-Finder DDU and a separate TF DDU event
	unsigned long long *start = (unsigned long long *)fuEvent[readyToGo];
	unsigned long long *end   = 0;
	enum {Header=1,Trailer=2};
	unsigned int eventStatus  = 0;
	for(int dduRecord=0; dduRecord<=tfDDUnumber; dduRecord++){
		unsigned long long word_0=0, word_1=0, word_2=0;
		size_t dduWordCount = 0;
		while( !end && dduWordCount<fuEventSize[readyToGo] ){
			unsigned long long *dduWord = start;

			while( dduWordCount<fuEventSize[readyToGo] ){
				word_0 =  word_1; // delay by 2 DDU words
				word_1 =  word_2; // delay by 1 DDU word
				word_2 = *dduWord;// current DDU word
				if( (word_2&0xFFFFFFFFFFFF0000LL)==0x8000000180000000LL ){
					if( eventStatus&Header ){ // Second header
						word_2 = word_1;
						end = dduWord;
						break;
					}
					start = dduWord;
				}
				if( (word_0&0xFFFFFFFFFFFF0000LL)==0x8000FFFF80000000LL ){
					eventStatus |= Trailer;
					end = ++dduWord;
					break;
				}
				// Increase counters by one DDU word
				dduWord++;
				dduWordCount++;
			}
		}
		// If reach max length
		if( dduWordCount==fuEventSize[readyToGo] ){
			end = (unsigned long long *)(fuEvent[readyToGo]+fuEventSize[readyToGo]);
			break;
		}
	}
	// Include 0x5xxx preHeader if exists
	if( start>(unsigned long long *)fuEvent[readyToGo] && (*(start-1)&0xF000000000000000LL)==0x5000000000000000LL ) start--;

	// If Track-Finder DDU was in readout
	if( tfDDUnumber>=0 ){
	// Cut out Track-Finder DDU from the buffer
		if( !end ) throw cms::Exception("CSCFileReader|lookingForTF")<<" Sanity check failed (end==0)! Should never happen";

		FEDRawData& tfRawData = data->FEDData(FEDNumbering::MINCSCTFFEDID);
		tfRawData.resize((end-start)*sizeof(unsigned long long));
		std::copy((unsigned char*)start,(unsigned char*)end,tfRawData.data());

		// Create a new buffer from everything before and after TF DDU
		unsigned short *event = tmpBuf;
		memcpy(event,fuEvent[readyToGo],((unsigned short*)start-fuEvent[readyToGo])*sizeof(unsigned short));
		event += ((unsigned short*)start-fuEvent[readyToGo]);
		memcpy(event,end,(fuEvent[readyToGo]+fuEventSize[readyToGo]-(unsigned short*)end)*sizeof(unsigned short));
		event += fuEvent[readyToGo]+fuEventSize[readyToGo]-(unsigned short*)end;
		FEDRawData& fedRawData = data->FEDData(FEDNumbering::MINCSCFEDID);
		fedRawData.resize((fuEventSize[readyToGo]-((unsigned short*)end-(unsigned short*)start))*sizeof(unsigned short));
		std::copy((unsigned char*)tmpBuf,(unsigned char*)event,fedRawData.data());
	} else {
		FEDRawData& fedRawData = data->FEDData(FEDNumbering::MINCSCFEDID);
		fedRawData.resize((fuEventSize[readyToGo])*sizeof(unsigned short));
		std::copy((unsigned char*)fuEvent[readyToGo],(unsigned char*)(fuEvent[readyToGo]+fuEventSize[readyToGo]),fedRawData.data());
	}

	currentL1A[readyToGo] = readFU(readyToGo,fuEvent[readyToGo],fuEventSize[readyToGo]);

	return eventNumber;
}

int CSCFileReader::fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection *& data){
	data = new FEDRawDataCollection();

	int runNumber   = 0; // Unknown at the level of EMu local DAQ
	int eventNumber =-1; // Will determine below

	if( !nActiveFUs && nActiveRUIs ){
		eventNumber = buildEventFromRUIs(data);
	} else {
		eventNumber = nextEventFromFUs(data);
	}

	if( eventNumber<0 ) return false;

	eID = edm::EventID(runNumber,1U,eventNumber);

	return true;
}

#undef nRUIs
#undef nFUs
