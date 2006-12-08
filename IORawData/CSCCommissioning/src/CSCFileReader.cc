#include "CSCFileReader.h"
#include <errno.h>
#include <stdlib.h>
#include <string>

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include <DataFormats/Common/interface/EventID.h>
#include <DataFormats/Common/interface/Timestamp.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <vector>
#include <string>
#include <iosfwd>
#include <sstream>
#include <iostream>
#include <algorithm>

#define nRUIs 10

CSCFileReader::CSCFileReader(const edm::ParameterSet& pset):DaqBaseReader(){
	// Get list of input files from .cfg file
	for(int rui=0; rui<nRUIs; rui++){
		std::ostringstream name;
		name<<"RUI0"<<rui<<std::ends;

		fileNames[rui] = pset.getUntrackedParameter< std::vector<std::string> >(name.str().c_str());
		currentFile[rui] = fileNames[rui].begin();

		if( currentFile[rui] != fileNames[rui].end() ){
			try {
				RUI[rui].open(currentFile[rui]->c_str());
			} catch ( std::runtime_error err ){
				throw cms::Exception("InputFileMissing ")<<"CSCFileReader: "<<err.what()<<" (errno="<<errno<<")";
			}
		}

		// Filter out possible corruptions
		RUI[rui].reject(FileReaderDDU::DDUoversize|FileReaderDDU::FFFF|FileReaderDDU::Unknown);
		// Do not select anything in particular
		RUI[rui].select(0);

		currentL1A[rui] = -1;
	}

	for(int fed=FEDNumbering::getCSCFEDIds().first; fed<=FEDNumbering::getCSCFEDIds().second; fed++){
		std::ostringstream name;
		name<<"FED"<<fed<<std::ends;
		std::vector<std::string> rui_list = pset.getUntrackedParameter< std::vector<std::string> >(name.str().c_str());
		for(std::vector<std::string>::const_iterator rui=rui_list.begin(); rui!=rui_list.end(); rui++)
			FED[fed].push_back((unsigned int)atoi(rui->c_str()+rui->length()-1));
	}
	for(int fed=FEDNumbering::getCSCTFFEDIds().first; fed<=FEDNumbering::getCSCTFFEDIds().second; fed++){
		std::ostringstream name;
		name<<"FED"<<fed<<std::ends;
		std::vector<std::string> rui_list = pset.getUntrackedParameter< std::vector<std::string> >(name.str().c_str());
		for(std::vector<std::string>::const_iterator rui=rui_list.begin(); rui!=rui_list.end(); rui++)
			FED[fed].push_back((unsigned int)atoi(rui->c_str()+rui->length()-1));
	}

	expectedNextL1A = -1;
}

int CSCFileReader::readEvent(int rui, const unsigned short* &buf, size_t &length){
	if( currentFile[rui] == fileNames[rui].end() ) return -1;
	do {
		try {
			length = RUI[rui].next(buf);
		} catch ( std::runtime_error err ){
			throw cms::Exception("EndOfStream")<<"CSCFileReader: "<<err.what()<<" (errno="<<errno<<")";
		}
		if( length==0 ){ // end of file, try next one
			if( ++currentFile[rui] != fileNames[rui].end() ){
				try {
					RUI[rui].open(currentFile[rui]->c_str());
				} catch ( std::runtime_error err ){
					throw cms::Exception("InputFileMissing ")<<"CSCFileReader: "<<err.what()<<" (errno="<<errno<<")";
				}
			} else return -1;
		}
	} while( length==0 );
	return buf[2]|((buf[3]&0xFF)<<16);
}

bool CSCFileReader::fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection *& data){
	data = new FEDRawDataCollection();

	// Event buffer and its length for every RUI
	const unsigned short *buf[nRUIs];
	size_t length[nRUIs];
	bzero(length,sizeof(length));

	// Read next event from RUIs
	for(int rui=0; rui<nRUIs; rui++){
		// read event from the RUI only in two cases:
		//     1) it is readable (currentL1A>0) and we expect next event from the RUI
		//     2) it is first time (expectedNextL1A<0)
		if((currentL1A[rui]>0 && currentL1A[rui]<expectedNextL1A) || expectedNextL1A<0 )
			currentL1A[rui] = readEvent(rui,buf[rui],length[rui]);
	}

	int runNumber   = 0; // Unknown at the level of EMu local DAQ
	int eventNumber =-1; // Will determine below

	// Select lowest L1A from all RUIs and don't expect next event from RUIs that currently hold higher L1A
	for(int rui=0; rui<nRUIs; rui++)
		if( currentL1A[rui]>0 && eventNumber<currentL1A[rui] ) eventNumber=currentL1A[rui];
	// No readable RUIs => fall out
	if( eventNumber<0 ) return false;
	// Expect next event to be incremented by 1 wrt. to the current event
	expectedNextL1A = eventNumber+1;

	eID = edm::EventID(runNumber,eventNumber);

	for(std::map<unsigned int,std::list<unsigned int> >::const_iterator fed=FED.begin(); fed!=FED.end(); fed++)
		if( fed->first<(unsigned int)FEDNumbering::getCSCTFFEDIds().first ){
			// Now let's pretend that DDU data were wrapped with DCC Header (2 64-bit words) and Trailer (2 64-bit words):
			unsigned short dccBuf[200000*nRUIs+4*4], *dccCur=dccBuf;
			dccCur[3] = 0x5000; dccCur[2] = 0x0000; dccCur[1] = 0x0000; dccCur[0] = 0x005F; // Fake DCC Header 1
			dccCur[7] = 0xD900; dccCur[6] = 0x0000; dccCur[5] = 0x0000; dccCur[4] = 0x0017; // Fake DCC Header 2
			dccCur += 8;

			for(std::list<unsigned int>::const_iterator rui=fed->second.begin(); rui!=fed->second.end(); rui++)
				if( currentL1A[*rui]==eventNumber ){
					if(dccCur-dccBuf+length[*rui]>=200000*nRUIs+8) throw cms::Exception("OutOfBuffer")<<"CSCFileReader: Event size exceeds maximal size allowed!";
					memcpy(dccCur,buf[*rui],length[*rui]*sizeof(unsigned short));
					dccCur += length[*rui];
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
				fedRawData.resize(length[*rui]*sizeof(unsigned short));
				std::copy((unsigned char*)buf[*rui],(unsigned char*)(buf[*rui]+length[*rui]),fedRawData.data());
			}
		}

	return true;
}

#undef nRUIs
