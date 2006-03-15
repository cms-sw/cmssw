#include "CSCFileReader.h"
#include "FileReaderDDU.h"
#include <errno.h>
#include <string>

#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include <DataFormats/Common/interface/EventID.h>
#include <DataFormats/Common/interface/Timestamp.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <string>
#include <iosfwd>
#include <iostream>
#include <algorithm>
   
using namespace std;
using namespace edm;

FileReaderDDU ___ddu;

CSCFileReader::CSCFileReader(const edm::ParameterSet& pset):DaqBaseReader(){
	// Following code is stolen from IORawData/DTCommissioning
	const std::string & filename = pset.getParameter<std::string>("fileName");
	try {
		___ddu.open(filename.c_str());
	} catch ( std::runtime_error err ){
		throw cms::Exception("InputFileMissing")<<"CSCFileReader: "<<err.what()<<" (errno="<<errno<<")";
	}
	// Filter out possible corruptions
	___ddu.reject(FileReaderDDU::DDUoversize|FileReaderDDU::FFFF|FileReaderDDU::Unknown);
	// Do not select anything in particular
	___ddu.select(0);
}

bool CSCFileReader::fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection& data){
	// Event buffer and its length
	const unsigned short *dduBuf=0;
	size_t length=0;

	try {
		// Read DDU record
		length = ___ddu.next(dduBuf);
	} catch ( std::runtime_error err ){
		throw cms::Exception("EndOfStream")<<"CSCFileReader: "<<err.what()<<" (errno="<<errno<<")";
	}

	if(!length) return false;

	int runNumber   = 0; // Unknown at the level of EMu local DAQ
	int eventNumber = dduBuf[2] | ((dduBuf[3]&0x00FF)<<16); // L1A Number
	eID = EventID(runNumber,eventNumber);

	// Now let's pretend that DDU data were wrapped with DCC Header (2 64-bit words) and Trailer (2 64-bit words):
	unsigned short dccBuf[200000+4*4], *dccHeader=dccBuf, *dccTrailer=dccBuf+4*2+length;
	memcpy(dccBuf+4*2,dduBuf,length*sizeof(unsigned short));
	dccHeader[3] = 0x5000; dccHeader[2] = 0x0000; dccHeader[1] = 0x0000; dccHeader[0] = 0x005F; // Fake DCC Header 1
	dccHeader[7] = 0xD900; dccHeader[6] = 0x0000; dccHeader[5] = 0x0000; dccHeader[4] = 0x0017; // Fake DCC Header 2
	dccTrailer[3] = 0xEF00; dccTrailer[2] = 0x0000; dccTrailer[1] = 0x0000; dccTrailer[0] = 0x0000; // Fake DCC Trailer 2
	dccTrailer[7] = 0xAF00; dccTrailer[6] = 0x0000; dccTrailer[5] = 0x0000; dccTrailer[4] = 0x0007; // Fake DCC Trailer 2
	length+=4*4;

    // The FED ID
    FEDRawData& fedRawData = data.FEDData( FEDNumbering::getCSCFEDIds().first );
    fedRawData.resize(length*sizeof(unsigned short));

	copy(reinterpret_cast<unsigned char*>(dccBuf),
		 reinterpret_cast<unsigned char*>(dccBuf+length), fedRawData.data());

	return true;
}
