// Last commit: $Id: $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::FedDescriptions& SiStripConfigDb::getFedDescriptions() {

  feds_.clear(); 
  
  if ( ( !dbParams_.usingDbCache_ && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache_ && !databaseCache(__func__) ) ) { return feds_; }

  try {

    if ( !dbParams_.usingDbCache_ ) { 

      deviceFactory(__func__)->setUsingStrips( usingStrips_ );
      int16_t major = dbParams_.fedMajor_; 
      int16_t minor = dbParams_.fedMinor_; 
      if ( dbParams_.fedMajor_ == 0 && 
	   dbParams_.fedMinor_ == 0 ) {
	major = -1; //@@ "current state" for fed factory!
	minor = -1; //@@ "current state" for fed factory!
      }
      feds_ = *( deviceFactory(__func__)->getFed9UDescriptions( dbParams_.partitions_.front(), 
								major, 
								minor ) );

    } else {

#ifdef USING_DATABASE_CACHE
      FedDescriptions* tmp = databaseCache(__func__)->getFed9UDescriptions();
      if ( tmp ) { feds_ = *tmp; }
      else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to FedDescriptions vector!";
      }
#endif

    }

  } catch (... ) { handleException( __func__ ); }
  
  // Debug 
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]" 
     << " Found " << feds_.size() 
     << " FED descriptions"; 
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.inputFedXml_.size() << " 'fed.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache_ )  { ss << " in database partition '" << dbParams_.partitions_.front() << "'"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory_ << "'"; } }
  if ( feds_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }
  
  return feds_;

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadFedDescriptions( bool new_major_version ) { 

  if ( dbParams_.usingDbCache_ ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]" 
      << " Using database cache! No uploads allowed!"; 
    return;
  }

  if ( !deviceFactory(__func__) ) { return; }
  
  if ( feds_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached FED descriptions, therefore no upload!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  try { 
    deviceFactory(__func__)->setFed9UDescriptions( feds_,
						   dbParams_.partitions_.front(),
						   (uint16_t*)(&dbParams_.fedMajor_), 
						   (uint16_t*)(&dbParams_.fedMinor_),
						   (new_major_version?1:0) ); 
  } catch (...) { handleException( __func__ ); }

  allowCalibUpload_ = true;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::createFedDescriptions( const SiStripFecCabling& fec_cabling ) {
  
  // Clear cache
  feds_.clear();
  
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
      feds_.push_back( f );
    } catch(...) {
      stringstream ss; 
      ss << "Problems creating description for FED id " << *ifed;
      handleException( __func__, ss.str() );
    }
  } 
  
  if ( feds_.empty() ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " No FED connections created!";
  }
  
}

// -----------------------------------------------------------------------------
/** */ 
const vector<uint16_t>& SiStripConfigDb::getFedIds() {
  
  fedIds_.clear();

  if ( feds_.empty() ) {
    bool using_strips = usingStrips_;
    if ( factory_ ) { factory_->setUsingStrips( false ); }
    getFedDescriptions();
    if ( factory_ ) { factory_->setUsingStrips( using_strips ); }
  }
  
  FedDescriptions::iterator ifed = feds_.begin();
  for ( ; ifed != feds_.end(); ifed++ ) { 
    if ( !(*ifed) ) {
      edm::LogError(mlCabling_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " NULL pointer to FedConnection!";
      continue;
    }
    fedIds_.push_back( (*ifed)->getFedId() );
  }
  
  if ( fedIds_.empty() ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " No FED ids found!"; 
  }
  
  return fedIds_;

}

