#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDRawDataToEvent.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CondFormats/DataRecord/interface/SiStripReadoutCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripReadoutCabling.h"
#include "Fed9UUtils.hh"
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Constructor
SiStripFEDRawDataToEvent::SiStripFEDRawDataToEvent( const edm::ParameterSet& pset ) :
  eventCounter_(0),
  fedReadoutMode_("VIRGIN_RAW"),
  verbosity_(0)
{
  std::cout << "[SiStripFEDRawDataToEvent::SiStripFEDRawDataToEvent]"
	    << " Constructing object..." << std::endl;
  
  // provide seed for random number generator
  srand( time( NULL ) ); 

  // Set configurables
  verbosity_ = pset.getParameter<int>("Verbosity");
  fedReadoutMode_ = pset.getParameter<std::string>("FedReadoutMode");

  // Define EDProduct type
  produces<FEDRawDataCollection>();
  
}

// -----------------------------------------------------------------------------
// Destructor
SiStripFEDRawDataToEvent::~SiStripFEDRawDataToEvent() {
  std::cout << "[SiStripFEDRawDataToEvent::~SiStripFEDRawDataToEvent]"
	    << " Destructing object..." << std::endl;
}

// -----------------------------------------------------------------------------
// Produces a FEDRawDataCollection
void SiStripFEDRawDataToEvent::produce( edm::Event& iEvent, 
					const edm::EventSetup& iSetup ) {
  
  // Some debug
  eventCounter_++; 
  if (verbosity_>0) std::cout << "[SiStripFEDRawDataToEvent::produce] "
 			      << "event number: " 
			      << eventCounter_ << std::endl;
  
  // Retrieve readout cabling map from EventSetup
  edm::ESHandle<SiStripReadoutCabling> cabling;
  iSetup.get<SiStripReadoutCablingRcd>().get( cabling );

  // Create EDProduct, a FEDRawDataCollection
  std::auto_ptr<FEDRawDataCollection> collection( new FEDRawDataCollection );

  // Some debug
  unsigned int ndigis = 0;
  unsigned int nchans = 0;

  // Retrieve FED ids and iterate through FEDs
  const std::vector<unsigned short> feds = cabling->getFEDs();
  std::vector<unsigned short>::const_iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    // Temporary container for ADC values
    std::vector<unsigned short> adc(96*256,0);
    for ( unsigned short ichan = 0; ichan < 96; ichan++ ) {
      // Only create digis for channels with non-zero DetId
      pair<unsigned int, unsigned short> apv_pair_id = cabling->getAPVPair( *ifed, ichan );
      if ( !apv_pair_id.first ) { continue; }
      // Channel counter
      nchans++; 
      // Calculate occupancy (1% *average* occupancy per chan)
      unsigned int ndigi = rand() % 5; //@@ flat distr!
      std::vector<int> strips; strips.reserve(ndigi);
      std::vector<int>::iterator iter;
      // Loop through "digis" and write adc/strip info to temp container
      unsigned int idigi = 0;
      while ( idigi < ndigi ) {
	int value = (rand() % 256) + 1; // adc value
	int strip = rand() % 256; // strip pos 
	iter = find( strips.begin(), strips.end(), strip );
	if ( iter == strips.end() ) { 
	  strips.push_back( strip ); 
	  adc[ichan*256+strip] = value;
	  ndigis++;
	  idigi++;
	}
      }
    }
    // Create appropriate object depending on FED readout mode
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
    // Create Fed9UBufferGenerator object (that uses Fed9UBufferCreator)
    Fed9U::Fed9UBufferGenerator generator( creator );
    generator.generateFed9UBuffer( adc );
    // Retrieve FEDRawData struct from collection
    FEDRawData& fed_data = collection->FEDData( *ifed ); 
    int nbytes = generator.getBufferSize() * 4;
    fed_data.resize( nbytes );
    // Copy FED buffer to FEDRawData struct 
    unsigned char* chars = const_cast<unsigned char*>( fed_data.data() );
    unsigned int* ints = reinterpret_cast<unsigned int*>( chars );
    generator.getBuffer( ints );
    if ( creator ) { delete creator; }
  }
  
  // Attach FEDRawDataCollection to Event
  iEvent.put( collection );
  
  if ( verbosity_ > 0 && nchans ) { 
    std::cout << "[WriteDummyDigisToEvent::produce]"
	      << " Generated " << ndigis 
	      << " digis for " << nchans
	      << " channels with an average occupancy of " 
	      << std::dec << std::setprecision(2)
	      << ( 100. / 256. ) * (float)ndigis / (float)nchans << " %"
	      << std::endl;
  }
  
}

//#include "PluginManager/ModuleDef.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
//DEFINE_FWK_MODULE(SiStripFEDRawDataToEvent)
