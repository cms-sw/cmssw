// Last commit: $Id: SiStripConfigDb.cc,v 1.9 2006/06/23 09:42:23 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/SiStripConfigDb.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::FedDescriptions& SiStripConfigDb::getFedDescriptions() {
  string method = "SiStripConfigDb::getFedDescriptions";
  edm::LogInfo(errorCategory_) << "["<<method<<"]"
			       << " Retrieving FED descriptions...";

  // If reset flag set, return contents of local cache
  if ( !resetFeds_ ) { return feds_; }

  try {
    deviceFactory(method)->setUsingStrips( usingStrips_ );
    feds_ = *( deviceFactory(method)->getFed9UDescriptions( partition_.name_, 
							    partition_.major_, 
							    partition_.minor_ ) );
    resetFeds_ = false;
  }
  catch (... ) {
    handleException( method );
  }
  
  stringstream ss; 
  if ( feds_.empty() ) {
    ss << "["<<method<<"]"
       << " No FED descriptions found";
    if ( !usingDb_ ) { ss << " in " << inputFedXml_.size() << " 'fed.xml' file(s)"; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogError(errorCategory_) << ss.str();
    throw cms::Exception(errorCategory_) << ss.str();
  } else {
    ss << "["<<method<<"]"
       << " Found " << feds_.size() << " FED descriptions";
    if ( !usingDb_ ) { ss << " in " << inputFedXml_.size() << " 'fed.xml' file(s)"; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogInfo(errorCategory_) << ss.str();
  }
  
  return feds_;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::resetFedDescriptions() {
  feds_.clear();
  resetFeds_ = true;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadFedDescriptions( bool new_major_version ) { //@@ this ok???
  string method = "SiStripConfigDb::uploadFedDescriptions";

  try { 
    if ( !usingDb_ ) {
      //deviceFactory(method)->addFedOutputFileName( "/tmp/fec.xml" ); //@@ ???
    }
    SiStripConfigDb::FedDescriptions::iterator ifed = feds_.begin();
    for ( ; ifed != feds_.end(); ifed++ ) {
      deviceFactory(method)->setFed9UDescription( **ifed, 
						  (uint16_t*)(&partition_.major_), 
						  (uint16_t*)(&partition_.minor_),
						  (new_major_version?1:0) );
    }
  }
  catch (...) { 
    handleException( method ); 
  }
  
}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::FedDescriptions& SiStripConfigDb::createFedDescriptions( const SiStripFecCabling& fec_cabling ) {
  string method = "SiStripConfigDb::createFedDescriptions";
  
  // Static container
  static FedDescriptions static_fed_descriptions;
  static_fed_descriptions.clear();
  
  // Create FED cabling from FEC cabling
  vector<FedChannelConnection> conns;
  fec_cabling.connections( conns );
  SiStripFedCabling* fed_cabling = new SiStripFedCabling( conns );

  // Retrieve and iterate through FED ids
  vector<uint16_t>::const_iterator ifed = fed_cabling->feds().begin();
  for ( ; ifed != fed_cabling->feds().end(); ifed++ ) {
    // Create FED description
    try {
      Fed9U::Fed9UDescription* f = new Fed9U::Fed9UDescription();
      f->setFedId( *ifed );
      f->setFedHardwareId( *ifed );
      Fed9U::Fed9UAddress addr;
      for ( uint32_t i = 0; i < Fed9U::APVS_PER_FED; i++ ) {
	addr.setFedApv(i);
	vector<Fed9U::u32> pedestals(128,100);
	vector<bool> disableStrips(128,false);
	vector<Fed9U::u32> highThresholds(128,50);
	vector<Fed9U::u32> lowThresholds(128,20);
	vector<Fed9U::Fed9UStripDescription> apvStripDescription(128);
	for ( uint32_t j = 0; j < Fed9U::STRIPS_PER_APV; j++) {
	  apvStripDescription[j].setPedestal(pedestals[j]);
	  apvStripDescription[j].setDisable(disableStrips[j]);
	  apvStripDescription[j].setLowThreshold(lowThresholds[j]);
	  apvStripDescription[j].setHighThreshold(highThresholds[j]);
	}
	f->getFedStrips().setApvStrips (addr, apvStripDescription);
      }
      static_fed_descriptions.push_back( f );
    } catch(...) {
      stringstream ss; 
      ss << "Problems creating description for FED id " << *ifed;
      handleException( method, ss.str() );
    }
  } 
  
  if ( static_fed_descriptions.empty() ) {
    stringstream ss;
    ss << "["<<method<<"] No FED connections created!";
    edm::LogError(errorCategory_) << ss.str() << "\n";
    //throw cms::Exception(errorCategory_) << ss.str() << "\n";
  }
  
  return static_fed_descriptions;

}

// -----------------------------------------------------------------------------
/** */ 
const vector<uint16_t>& SiStripConfigDb::getFedIds() {
  
  static vector<uint16_t> fed_ids;
  fed_ids.clear();

  getFedDescriptions();
  FedDescriptions::iterator ifed = feds_.begin();
  for ( ; ifed != feds_.end(); ifed++ ) { 
    fed_ids.push_back( (*ifed)->getFedId() );
  }
  if ( fed_ids.empty() ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::getFedIds]"
			      << " No FED ids found!"; 
  }
  return fed_ids;
}
