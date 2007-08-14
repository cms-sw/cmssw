/** \file
 *
 *  $Date: 2007/04/19 13:28:46 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include <IORawData/DaqSource/interface/DaqReaderPluginFactory.h>

#include "DaqFakeReader.h"
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <CLHEP/Random/RandGauss.h>

#include <cmath>
   
using namespace std;
using namespace edm;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
DaqFakeReader::DaqFakeReader(const edm::ParameterSet& pset) 
  : runNum(1)
  , eventNum(0)
  , empty_events(pset.getUntrackedParameter<bool>("emptyEvents",false)) 
{
  // mean = pset.getParameter<float>("mean");
}

//______________________________________________________________________________
DaqFakeReader::~DaqFakeReader()
{
  
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
bool DaqFakeReader::fillRawData(EventID& eID,
				Timestamp& tstamp, 
				FEDRawDataCollection*& data)
{
  // a null pointer is passed, need to allocate the fed collection
  data=new FEDRawDataCollection();
  
  if(!empty_events)
    {
      // FIXME
      float meansize = 1024;
      float width = 1024;
      
      // Fill the EventID
      eID = EventID(runNum,eventNum);
      eventNum++;
      // FIXME:
      
      fillFEDs(FEDNumbering::getSiPixelFEDIds(), eID, tstamp, *data, meansize, width);
      fillFEDs(FEDNumbering::getSiStripFEDIds(), eID, tstamp, *data, meansize, width);
      
      fillFEDs(FEDNumbering::getDTFEDIds(), eID, tstamp, *data, meansize, width);
      fillFEDs(FEDNumbering::getCSCFEDIds(), eID, tstamp, *data, meansize, width);
      fillFEDs(FEDNumbering::getRPCFEDIds(), eID, tstamp, *data, meansize, width);
      
      fillFEDs(FEDNumbering::getEcalFEDIds(), eID, tstamp, *data, meansize, width);
      fillFEDs(FEDNumbering::getHcalFEDIds(), eID, tstamp, *data, meansize, width);
    }
  return true;
}


//______________________________________________________________________________
void DaqFakeReader::fillFEDs(const pair<int,int>& fedRange,
			     EventID& eID,
			     Timestamp& tstamp, 
			     FEDRawDataCollection& data,
			     float meansize,
			     float width)
{
  
  // FIXME: last ID included?
  for (int fedId = fedRange.first; fedId <= fedRange.second; ++fedId ) {
    
    // Generate size...
    float logsiz = RandGauss::shoot(std::log(meansize),
				    std::log(meansize)-std::log(width/2.));
    size_t size = int(std::exp(logsiz));
    size -= size % 8; // all blocks aligned to 64 bit words

    FEDRawData& feddata = data.FEDData(fedId);
    // Allocate space for header+trailer+payload
    feddata.resize(size+16); 

    // Generate header
    FEDHeader::set(feddata.data(),
	       1,             // Trigger type
	       eID.event(),   // LV1_id
	       0,             // BX_id
	       fedId);        // source_id

    // Payload = all 0s...

    // Generate trailer
    int crc = 0; // FIXME : get CRC
    FEDTrailer::set(feddata.data()+8+size,
		    size/8+2, // in 64 bit words!!!
		    crc,
		    0,        // Evt_stat
		    0);       // TTS bits
  }  
}

// DaqFakeReader as a SEAL plugin

DEFINE_EDM_PLUGIN (DaqReaderPluginFactory, DaqFakeReader, "DaqFakeReader");
