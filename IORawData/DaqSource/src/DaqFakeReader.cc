
/* \file DaqFakeReader.cc
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include "DaqFakeReader.h"
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include <FWCore/EDProduct/interface/EventID.h>
#include <FWCore/EDProduct/interface/Timestamp.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <CLHEP/Random/RandGauss.h>
   
using namespace std;
using namespace edm;
using namespace raw;


DaqFakeReader::DaqFakeReader(const edm::ParameterSet& pset){
  // mean = pset.getParameter<float>("mean");
}


DaqFakeReader::~DaqFakeReader(){}


bool DaqFakeReader::fillRawData(EventID& eID,
				Timestamp& tstamp, 
				FEDRawDataCollection& data){

  float meansize = 1024;
  float width = 1024;


  fillFEDs(FEDNumbering::getSiPixelFEDIds(), eID, tstamp, data, meansize, width);
  fillFEDs(FEDNumbering::getSiStripFEDIds(), eID, tstamp, data, meansize, width);

  fillFEDs(FEDNumbering::getMuBarFEDIds(), eID, tstamp, data, meansize, width);
  fillFEDs(FEDNumbering::getMuEndFEDIds(), eID, tstamp, data, meansize, width);
  fillFEDs(FEDNumbering::getMRpcFEDIds(), eID, tstamp, data, meansize, width);

  fillFEDs(FEDNumbering::getEcalBarrelFEDIds(), eID, tstamp, data, meansize, width);
  fillFEDs(FEDNumbering::getEcalEndcapFEDIds(), eID, tstamp, data, meansize, width);
  fillFEDs(FEDNumbering::getHcalFEDIds(), eID, tstamp, data, meansize, width);

  return true;
}

void DaqFakeReader::fillFEDs(const pair<int,int>& fedRange,
			     EventID& eID,
			     Timestamp& tstamp, 
			     FEDRawDataCollection& data,
			     float meansize,
			     float width){

  // FIXME: last ID included?
  for (int fedId = fedRange.first; fedId <= fedRange.second; ++fedId ) {
    
    // Generate size...
    float logsiz = RandGauss::shoot(log(meansize),log(meansize)-log(width/2.));
    size_t size = int(exp(logsiz));
    size -= size % 8; // all blocks aligned to 64 bit words

    FEDRawData& feddata = data.FEDData(fedId);
    // Allocate space for header+trailer+payload
    feddata.resize(size+16); 

    // Generate header
    FEDHeader header(feddata.data());
    header.set(1,             // Trigger type
	       eID.event(),   // LV1_id
	       0,             // BX_id
	       fedId);        // source_id

    // Payload = all 0s...

    // Generate trailer
    FEDTrailer trailer(feddata.data()+8+size);
    int crc = 0; // FIXME : get CRC
    trailer.set(size/8+2, // in 64 bit words!!!
		crc,
		0,        // Evt_stat
		0);       // TTS bits
  }  
}

