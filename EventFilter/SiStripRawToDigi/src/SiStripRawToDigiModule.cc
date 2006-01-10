#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiModule.h"
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "CondFormats/DataRecord/interface/SiStripReadoutCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripReadoutCabling.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripRawToDigi.h"
#include <cstdlib>

// -----------------------------------------------------------------------------
// Constructor
SiStripRawToDigiModule::SiStripRawToDigiModule( const edm::ParameterSet& pset ) :
  rawToDigi_(0),
  eventCounter_(0),
  verbosity_(0)
{
  if ( verbosity_>0 ) {
    std::cout << "[SiStripRawToDigiModule::SiStripRawToDigiModule]"
	      << " Constructing object..." << std::endl;
  }
  // Set some private data members 
  verbosity_ = pset.getParameter<int>("Verbosity");
  // Create instance of RawToDigi converter
  rawToDigi_ = new SiStripRawToDigi( verbosity_ );
  rawToDigi_->fedReadoutMode( pset.getParameter<std::string>("FedReadoutMode") );
  rawToDigi_->fedReadoutPath( pset.getParameter<std::string>("FedReadoutPath") );
  // Define EDProduct type
  produces<StripDigiCollection>();
}

// -----------------------------------------------------------------------------
// Destructor
SiStripRawToDigiModule::~SiStripRawToDigiModule() {
  if ( verbosity_>0 ) {
    std::cout << "[SiStripRawToDigiModule::~SiStripRawToDigiModule]"
	      << " Destructing object..." << std::endl;
  }
  if ( rawToDigi_ ) delete rawToDigi_;
}

// -----------------------------------------------------------------------------
// Produces a StripDigiCollection
void SiStripRawToDigiModule::produce( edm::Event& iEvent, 
				      const edm::EventSetup& iSetup ) {
  
  // Some debug
  eventCounter_++; 
  if (verbosity_>1) std::cout << "[SiStripRawToDigiModule::produce]"
 			      << " processing event number: " 
			      << eventCounter_ << std::endl;
  
  // Retrieve readout cabling map from EventSetup
  edm::ESHandle<SiStripReadoutCabling> cabling;
  iSetup.get<SiStripReadoutCablingRcd>().get( cabling );

  // Retrieve input data from Event, a FEDRawDataCollection
  edm::Handle<FEDRawDataCollection> buffers;
  iEvent.getByType( buffers );

  // Create EDProduct, a StripDigiCollection
  std::auto_ptr<StripDigiCollection> digis( new StripDigiCollection );

  // Use RawToDigi unpacker to fill FEDRawDataCollection
  //rawToDigi_->createDigis( cabling, buffers, digis );
  
  std::cout << "before" << std::endl;
  // Attach StripDigiCollection to Event
  iEvent.put( digis );
  std::cout << "after" << std::endl;
 
  if (verbosity_>2) std::cout << "[SiStripRawToDigiModule::produce]"
 			      << " exiting method... " << std::endl;
  
}
