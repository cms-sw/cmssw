#include "CalibFormats/SiStripObjects/interface/SiStripFedCablingESSource.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include <iostream>

// -----------------------------------------------------------------------------
//
SiStripFedCablingESSource::SiStripFedCablingESSource( const edm::ParameterSet& pset ) {
  std::cout << "[SiStripFedCablingESSource::SiStripFedCablingESSource]"
	    << " Constructing object..." << std::endl;

  if ( pset.getParameter<std::string>("Label") == "" ) { //@@ what is this?
    setWhatProduced( this );
  } else {
    setWhatProduced( this, pset.getParameter<std::string>("Label") );
  }
  findingRecord<SiStripFedCablingRcd>();

}

// -----------------------------------------------------------------------------
//
std::auto_ptr<SiStripFedCabling> SiStripFedCablingESSource::produce( const SiStripFedCablingRcd& ) { 
  std::cout << "[SiStripFedCablingESSource::produce]" << std::endl;
  
  SiStripFedCabling* cabling = makeFedCabling();
  
  if ( !cabling ) {
    std::cerr << "[SiStripFedCablingESSource::produce]"
	      << " Null SiStripFedCabling pointer!" << std::endl;
  }
  
  std::auto_ptr<SiStripFedCabling> ptr(cabling);
  return ptr;

}

// -----------------------------------------------------------------------------
//
void SiStripFedCablingESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&, 
						const edm::IOVSyncValue& iosv, 
						edm::ValidityInterval& oValidity ) {
  std::cout << "[SiStripFedCablingESSource::setIntervalFor]" << std::endl;
  
  edm::ValidityInterval infinity( iosv.beginOfTime(), iosv.endOfTime() );
  oValidity = infinity;
  
}

//#include "FWCore/Framework/interface/SourceFactory.h"
//DEFINE_FWK_MODULE(SiStripFedCablingESSource)
//DEFINE_FWK_MODULE(SiStripFedCablingBuilder) ???


