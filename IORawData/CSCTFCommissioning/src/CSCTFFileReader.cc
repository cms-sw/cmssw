#include <IORawData/CSCTFCommissioning/src/CSCTFFileReader.h>
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
#include <IORawData/CSCTFCommissioning/src/FileReaderSP.h>
#include <IORawData/CSCTFCommissioning/src/FileReaderSPNewFormat.h>

#include <string>
#include <iosfwd>
#include <iostream>
#include <algorithm>
   
using namespace std;
using namespace edm;

//FileReaderSP ___ddu;

CSCTFFileReader::CSCTFFileReader(const edm::ParameterSet& pset):DaqBaseReader()
{
  // Following code is stolen from IORawData/DTCommissioning
  const std::string dataformat = pset.getUntrackedParameter<std::string>("dataFormat","TestBeam");

  if(dataformat == "TestBeam")
    {
      ___ddu = new FileReaderSP();
    }
  if(dataformat == "Final")
    {
      ___ddu = new FileReaderSPNewFormat();
    }

  const std::string & filename = pset.getParameter<std::string>("fileName");
  std::cout << "Opening File: " << filename << std::endl;
  ___ddu->openFile(filename.c_str());
}

CSCTFFileReader::~CSCTFFileReader()
{
  delete ___ddu;
}

bool CSCTFFileReader::fillRawData(edm::EventID& eID, edm::Timestamp& tstamp, FEDRawDataCollection*& data){
  data = new FEDRawDataCollection();

  // Event buffer and its length
  size_t length=0;
	
  // Read DDU record
  ___ddu->readNextEvent();
  const unsigned short* dduBuf = reinterpret_cast<unsigned short*>(___ddu->data());
  length = ___ddu->dataLength();
  
  if(!length) {
    delete data; data=0;
    return false;
  }
  
  int runNumber   = 0; // Unknown at the level of EMu local DAQ
  int eventNumber =((dduBuf[2])&0x0FFF) | ((dduBuf[3]&0x0FFF)<<12); // L1A Number
  eID = EventID(runNumber,eventNumber);
  

  unsigned short dccBuf[200000+4*4];//, *dccHeader=dccBuf, *dccTrailer=dccBuf+4*2+(length/2);
  memcpy(dccBuf,dduBuf,length);

  // The FED ID
  FEDRawData& fedRawData = data->FEDData( FEDNumbering::getCSCFEDIds().first );
  int newlength = 0;
  if(length%8)
    {
      newlength = length + (8-(length%8));
    }
  else newlength = length;
  fedRawData.resize(newlength);

  copy(reinterpret_cast<unsigned char*>(dccBuf),
       reinterpret_cast<unsigned char*>(dccBuf)+length, fedRawData.data());

    return true;
}

