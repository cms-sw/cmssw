// Last commit: $Id: FedDescriptions.cc,v 1.2 2006/07/26 11:27:19 bainbrid Exp $
// Latest tag:  $Name: V00-01-02 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/FedDescriptions.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::FedDescriptions& SiStripConfigDb::getFedDescriptions() {
  edm::LogInfo(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]"
			     << " Retrieving FED descriptions...";

  if ( !deviceFactory(__FUNCTION__) ) { return feds_; }
  if ( !resetFeds_ ) { return feds_; }

  try {
    deviceFactory(__FUNCTION__)->setUsingStrips( usingStrips_ );
    feds_ = *( deviceFactory(__FUNCTION__)->getFed9UDescriptions( partition_.name_, 
								  partition_.major_, 
								  partition_.minor_ ) );
    resetFeds_ = false;
  }
  catch (... ) {
    handleException( __FUNCTION__ );
  }
  
  stringstream ss; 
  if ( feds_.empty() ) {
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " No FED descriptions found";
    if ( !usingDb_ ) { ss << " in " << inputFedXml_.size() << " 'fed.xml' file(s)"; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogError(logCategory_) << ss.str();
    throw cms::Exception(logCategory_) << ss.str();
  } else {
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " Found " << feds_.size() << " FED descriptions";
    if ( !usingDb_ ) { ss << " in " << inputFedXml_.size() << " 'fed.xml' file(s)"; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogInfo(logCategory_) << ss.str();
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

  if ( !deviceFactory(__FUNCTION__) ) { return; }
  
  try { 
    SiStripConfigDb::FedDescriptions::iterator ifed = feds_.begin();
    for ( ; ifed != feds_.end(); ifed++ ) {
      deviceFactory(__FUNCTION__)->setFed9UDescription( **ifed, 
							(uint16_t*)(&partition_.major_), 
							(uint16_t*)(&partition_.minor_),
							(new_major_version?1:0) );
    }
  }
  catch (...) { 
    handleException( __FUNCTION__ ); 
  }
  
}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::FedDescriptions& SiStripConfigDb::createFedDescriptions( const SiStripFecCabling& fec_cabling ) {
  
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
      handleException( __FUNCTION__, ss.str() );
    }
  } 
  
  if ( static_fed_descriptions.empty() ) {
    stringstream ss;
    ss << "[" << __PRETTY_FUNCTION__ << "] No FED connections created!";
    edm::LogError(logCategory_) << ss.str() << "\n";
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
