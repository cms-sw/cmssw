#include "EventFilter/SiStripRawToDigi/interface/SiStripUtility.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerSimAlgo/interface/CmsDigiTracker.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibTracker/SiStripConnectivity/interface/SiStripConnection.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Fed9UUtils.hh"

#include <vector>
#include <string>

using namespace std;
using namespace raw;

// -----------------------------------------------------------------------------
// constructor
SiStripUtility::SiStripUtility( const edm::EventSetup& iSetup ) :
  nDets_(0)
{
  cout << "[SiStripUtility] : constructing class..." << endl;
  // get geometry 
//   edm::eventsetup::ESHandle<CmsDigiTracker> pDD;
//   iSetup.get<TrackerDigiGeometryRecord>().get( pDD );
//   vector<GeomDetUnit*> dets = pDD->dets();
  nDets_ = 15000; // pDD->dets().size();
}

// -----------------------------------------------------------------------------
// destructor
SiStripUtility::~SiStripUtility() {
  cout << "[SiStripUtility] : destructing class..." << endl;
 }

// -----------------------------------------------------------------------------
//
void SiStripUtility::stripDigiCollection( StripDigiCollection& collection ) {
  cout << "[SiStripUtility::stripDigiCollection]" << endl;
  // loop through detectors
  for ( int idet = 0; idet < nDets_; idet++ ) { 
    // some random numbers
    int ndigi = rand()%20, strip = rand()%512, value = rand()%50;
    // temorary digi container
    vector<StripDigi> digis; 
    // create digis
    for ( int idigi = 0; idigi < ndigi; idigi++ ) { digis.push_back( StripDigi(strip,value) ); }
    // digi range
    StripDigiCollection::Range range = StripDigiCollection::Range( digis.begin(), digis.end() );
    // put digis in collection
    collection.put(range, idet);
  }
}

// -----------------------------------------------------------------------------
//
void SiStripUtility::fedRawDataCollection( FEDRawDataCollection& collection ) {
  cout << "[SiStripUtility::fedRawDataCollection]" << endl;

  // calculate number of FEDs for given number of detectors 
  unsigned int nFeds = nDets_/48 ? (nDets_/48)+1 : (nDets_/48);

  // loop over FEDs
  for ( unsigned int ifed = 0; ifed < nFeds; ifed++ ) { 

    // temp container for adc values
    vector<unsigned short> adc(96*256,0);
    // loop through FED channels
    for ( int ichan = 0; ichan < 96; ichan++ ) { 
      // some random numbers
      int ndigi = rand()%10, strip = rand()%256, value = rand()%50;
      // write adc values to temporary adc container
      for ( int idigi = 0; idigi < ndigi; idigi++ ) { adc[ichan*256+strip] = value; }
    }

    // instantiate appropriate buffer creator object depending on readout mode
    Fed9U::Fed9UBufferCreator* creator = 0;
    string readout_mode = "VIRGIN_RAW";
    if ( readout_mode == "SCOPE_MODE" ) {
      cout << "WARNING : SCOPE_MODE not implemented yet!" << endl;
    } else if ( readout_mode == "VIRGIN_RAW" ) {
      creator = new Fed9U::Fed9UBufferCreatorRaw();
    } else if ( readout_mode == "PROCESSED_RAW" ) {
      creator = new Fed9U::Fed9UBufferCreatorProcRaw();
  } else if ( readout_mode == "ZERO_SUPPRESSED" ) {
      creator = new Fed9U::Fed9UBufferCreatorZS();
    } else {
      cout << "WARNING : UNKNOWN readout mode"<<endl;
    }
 
    // create Fed9UBufferGenerator object (that uses Fed9UBufferCreator)
    Fed9U::Fed9UBufferGenerator generator( creator );
    // generate FED buffer using vector<unsigned short> that holds adc values
    generator.generateFed9UBuffer( adc );
    // retrieve FEDRawData struct from collection for appropriate fed
    FEDRawData& fed_data = collection.FEDData( ifed ); 
    // calculate size of FED buffer in units of bytes (unsigned char)
    int nbytes = generator.getBufferSize() * 4;
    // resize (public) "data_" member of struct FEDRawData
    (fed_data.data_).resize( nbytes );
    // copy FED buffer to struct FEDRawData using Fed9UBufferGenerator
    unsigned char* chars = const_cast<unsigned char*>( fed_data.data() );
    unsigned int* ints = reinterpret_cast<unsigned int*>( chars );
    generator.getBuffer( ints );
    
  }
}

// -----------------------------------------------------------------------------
//
void SiStripUtility::siStripConnection( SiStripConnection& connections ) {
  cout << "[SiStripUtility::siStripConnection]" << endl;
  // loop through detectors
  for ( int idet = 0; idet < nDets_; idet++ ) { 
    pair<unsigned short, unsigned short> fed; 
    pair<cms::DetId,unsigned short> det; 
    fed = pair<unsigned short, unsigned short>( (idet/48), idet%48 );
    det = pair<cms::DetId,unsigned short>( idet, 0 );
    connections.setPair( fed, det );
    fed = pair<unsigned short, unsigned short>( (idet/48), (idet%48)+1 );
    det = pair<cms::DetId,unsigned short>( idet, 1 );
    connections.setPair( fed, det );
  }
}
