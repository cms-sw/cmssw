#include "CSCFileReader.h"

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

#include <vector>
#include <string>
#include <iosfwd>
#include <iostream>
#include <algorithm>

#include "FileReaderDDU.h"
namespace ___CSC {
	FileReaderDDU ddu;
};

CSCFileReader::CSCFileReader(const edm::ParameterSet& pset):DaqBaseReader(){
	// Define type of the output data first: if DAQ - wrapp DDU buffer into fake DCC, if not - not
	if( pset.getUntrackedParameter<std::string>("dataType") != "TF" )
		dataType = DAQ;
	else 
		dataType = TF;

	// Get list of input files from .cfg file
	fileNames   = pset.getUntrackedParameter< std::vector<std::string> >("fileNames");
	currentFile = fileNames.begin();
	if( currentFile != fileNames.end() ){
		try {
			___CSC::ddu.open(currentFile->c_str());
		} catch ( std::runtime_error err ){
			throw cms::Exception("InputFileMissing ")<<"CSCFileReader: "<<err.what()<<" (errno="<<errno<<")";
		}
	} else  throw cms::Exception("NoFilesSpecified for CSCFileReader");
	// Filter out possible corruptions
	___CSC::ddu.reject(FileReaderDDU::DDUoversize|FileReaderDDU::FFFF|FileReaderDDU::Unknown);
	// Do not select anything in particular
	___CSC::ddu.select(0);
}

bool CSCFileReader::fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection& data){
	// Event buffer and its length
	const unsigned short *dduBuf=0;
	size_t length=0;

	try {
		// Read DDU record
		length = ___CSC::ddu.next(dduBuf);
	} catch ( std::runtime_error err ){
		throw cms::Exception("EndOfStream")<<"CSCFileReader: "<<err.what()<<" (errno="<<errno<<")";
	}

	if( length==0 ){ // end of file, try next one
		if( ++currentFile != fileNames.end() ){
			try {
				___CSC::ddu.open(currentFile->c_str());
			} catch ( std::runtime_error err ){
				throw cms::Exception("InputFileMissing ")<<"CSCFileReader: "<<err.what()<<" (errno="<<errno<<")";
			}
		} else return false;
	}

	int runNumber   = 0; // Unknown at the level of EMu local DAQ
	int eventNumber = dduBuf[2] | ((dduBuf[3]&0x00FF)<<16); // L1A Number
	eID = edm::EventID(runNumber,eventNumber);

	// Now let's pretend that DDU data were wrapped with DCC Header (2 64-bit words) and Trailer (2 64-bit words):
	unsigned short dccBuf[200000+4*4], *dccHeader=dccBuf, *dccTrailer=dccBuf+4*2+length;
	memcpy(dccBuf+4*2,dduBuf,length*sizeof(unsigned short));
	dccHeader[3] = 0x5000; dccHeader[2] = 0x0000; dccHeader[1] = 0x0000; dccHeader[0] = 0x005F; // Fake DCC Header 1
	dccHeader[7] = 0xD900; dccHeader[6] = 0x0000; dccHeader[5] = 0x0000; dccHeader[4] = 0x0017; // Fake DCC Header 2
	dccTrailer[3] = 0xEF00; dccTrailer[2] = 0x0000; dccTrailer[1] = 0x0000; dccTrailer[0] = 0x0000; // Fake DCC Trailer 2
	dccTrailer[7] = 0xAF00; dccTrailer[6] = 0x0000; dccTrailer[5] = 0x0000; dccTrailer[4] = 0x0007; // Fake DCC Trailer 2
	length+=4*4;

	// The FED ID
	if( dataType == DAQ ){
		FEDRawData& fedRawData = data.FEDData( FEDNumbering::getCSCFEDIds().first );
		fedRawData.resize(length*sizeof(unsigned short));
		std::copy((unsigned char*)dccBuf,(unsigned char*)(dccBuf+length),fedRawData.data());
	} else {
		FEDRawData& fedRawData = data.FEDData( FEDNumbering::getCSCTFFEDIds().first );
		fedRawData.resize((length-4*4)*sizeof(unsigned short));
		std::copy((unsigned char*)dduBuf,(unsigned char*)(dduBuf+length-4*4),fedRawData.data());
	}

	return true;
}
