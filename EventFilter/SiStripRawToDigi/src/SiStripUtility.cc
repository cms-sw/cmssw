#include "EventFilter/SiStripRawToDigi/interface/SiStripUtility.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerSimAlgo/interface/CmsDigiTracker.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibTracker/SiStripConnectivity/interface/SiStripConnection.h"
#include "DataFormats/DetId/interface/DetId.h"
//
#include "Fed9UUtils.hh"
//
#include <cstdlib>
#include <ctime>
#include <vector>

// -----------------------------------------------------------------------------
// constructor
SiStripUtility::SiStripUtility( const edm::EventSetup& iSetup ) :
  nDets_(0),
  fedReadoutMode_("ZERO_SUPPRESSED"),
  verbose_(false)
{
  if (verbose_) std::cout << "[SiStripUtility::SiStripUtility] "
			  << "constructing class..." << std::endl;
  //   // get geometry 
  //   edm::eventsetup::ESHandle<CmsDigiTracker> pDD;
  //   iSetup.get<TrackerDigiGeometryRecord>().get( pDD );
  //   std::vector<GeomDetUnit*> dets = pDD->dets();
  //   nDets_ = pDD->dets().size();
  nDets_ = 15000; 
  // provide seed for random number generator
  srand( time( NULL ) ); 
}

// -----------------------------------------------------------------------------
// destructor
SiStripUtility::~SiStripUtility() {
  if (verbose_) std::cout << "[SiStripUtility~SiStripUtility] : destructing class..." << std::endl;
 }


// -----------------------------------------------------------------------------
//
void SiStripUtility::siStripConnection( SiStripConnection& connections ) {
  if (verbose_) std::cout << "[SiStripUtility::siStripConnection]" << std::endl;
  // loop through detectors
  for ( int idet = 0; idet < nDets_; idet++ ) { 
    pair<unsigned short, unsigned short> fed; 
    pair<cms::DetId,unsigned short> det; 
    fed = pair<unsigned short, unsigned short>( (idet/48), 2*(idet%48) );
    det = pair<cms::DetId,unsigned short>( idet, 0 );
    connections.setPair( fed, det );
    fed = pair<unsigned short, unsigned short>( (idet/48), 2*(idet%48)+1 );
    det = pair<cms::DetId,unsigned short>( idet, 1 );
    connections.setPair( fed, det );
  }
}

// -----------------------------------------------------------------------------
//
void SiStripUtility::stripDigiCollection( StripDigiCollection& collection ) {
  if (verbose_) std::cout << "[SiStripUtility::stripDigiCollection]" << std::endl;
  // loop through detectors
  for ( int idet = 0; idet < nDets_; idet++ ) { 
    int ndigi = rand() % 16; // number of hits per det (~3% occupancy)
    std::vector<StripDigi> digis; // temp container
    // loop through and create digis
    for ( int idigi = 0; idigi < ndigi; idigi++ ) { 
      int strip = rand() % 512; // strip position of hit
      int value = rand() % 50; // adc value
      digis.push_back( StripDigi(strip,value) ); 
    }
    // digi range
    StripDigiCollection::Range range = StripDigiCollection::Range( digis.begin(), digis.end() );
    // put digis in collection
    collection.put(range, idet);
  }
}

// -----------------------------------------------------------------------------
//
void SiStripUtility::fedRawDataCollection( raw::FEDRawDataCollection& collection ) {
  if (verbose_) std::cout << "[SiStripUtility::fedRawDataCollection]" << std::endl;

  // calculate number of FEDs for given number of detectors 
  unsigned int nFeds = nDets_%48 ? (nDets_/48)+1 : (nDets_/48);

  // loop over FEDs
  for ( unsigned int ifed = 0; ifed < nFeds; ifed++ ) { 

    // temp container for adc values
    std::vector<unsigned short> adc(96*256,0);
    // loop through FED channels
    for ( int ichan = 0; ichan < 96; ichan++ ) { 
      // some random numbers
      int ndigi = rand() % 8; // number of hits per FED channel (~3% occupancy)
      // write adc values to temporary adc container
      for ( int idigi = 0; idigi < ndigi; idigi++ ) { 
	int strip = rand() % 256; // strip position of hit
	int value = rand() % 50; // adc value
	adc[ichan*256+strip] = value; 
      }
    }
    
    // instantiate appropriate buffer creator object depending on FED readout mode
    Fed9U::Fed9UBufferCreator* creator = 0;
    if ( fedReadoutMode_ == "SCOPE_MODE" ) {
      std::cout << "WARNING : SCOPE_MODE not implemented yet!" << std::endl;
    } else if ( fedReadoutMode_ == "VIRGIN_RAW" ) {
      creator = new Fed9U::Fed9UBufferCreatorRaw();
    } else if ( fedReadoutMode_ == "PROCESSED_RAW" ) {
      creator = new Fed9U::Fed9UBufferCreatorProcRaw();
    } else if ( fedReadoutMode_ == "ZERO_SUPPRESSED" ) {
      creator = new Fed9U::Fed9UBufferCreatorZS();
    } else {
      std::cout << "WARNING : UNKNOWN readout mode" << std::endl;
    }
 
    // create Fed9UBufferGenerator object (that uses Fed9UBufferCreator)
    Fed9U::Fed9UBufferGenerator generator( creator );
    // generate FED buffer using std::vector<unsigned short> that holds adc values
    generator.generateFed9UBuffer( adc );
    // retrieve raw::FEDRawData struct from collection for appropriate fed
    raw::FEDRawData& fed_data = collection.FEDData( ifed ); 
    // calculate size of FED buffer in units of bytes (unsigned char)
    int nbytes = generator.getBufferSize() * 4;
    // resize (public) "data_" member of struct raw::FEDRawData
    (fed_data.data_).resize( nbytes );
    // copy FED buffer to struct raw::FEDRawData using Fed9UBufferGenerator
    unsigned char* chars = const_cast<unsigned char*>( fed_data.data() );
    unsigned int* ints = reinterpret_cast<unsigned int*>( chars );
    generator.getBuffer( ints );
    
  }
}
