// Last commit: $Id: PiaResetDescriptions.cc,v 1.3 2006/08/31 19:49:41 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/PiaResetDescriptions.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::PiaResetDescriptions& SiStripConfigDb::getPiaResetDescriptions() {
  
  if ( !deviceFactory(__func__) ) { return piaResets_; }
  if ( !resetPiaResets_ ) { return piaResets_; }
  
  try { 
    deviceFactory(__func__)->getPiaResetDescriptions( partition_.name_, piaResets_ );
    resetPiaResets_ = false;
  } catch (...) { 
    handleException( __func__ ); 
  }
  
  if ( piaResets_.empty() ) {
    edm::LogError(mlConfigDb_) << "No PIA reset descriptions found!";
  }
  
  return piaResets_;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::resetPiaResetDescriptions() {
  piaResets_.clear();
  resetPiaResets_ = true;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadPiaResetDescriptions() {

  if ( !deviceFactory(__func__) ) { return; }
  
  try { 
    deviceFactory(__func__)->setPiaResetDescriptions( piaResets_, 
						      partition_.name_ );
  } catch (...) { 
    handleException( __func__ ); 
  }
  
}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::PiaResetDescriptions& SiStripConfigDb::createPiaResetDescriptions( const SiStripFecCabling& fec_cabling ) {
  
  // Container
  static PiaResetDescriptions static_pia_resets;
  static_pia_resets.clear();
  
  // Unique key (within a partition)
  keyType index; 
  
  // Iterate through control system, create descriptions and populate containers 
  for ( vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin(); icrate != fec_cabling.crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {

      // FEC hardware id (encodes crate + slot numbers)
      stringstream fec_hardware_id; 
      fec_hardware_id << setw(4) << setfill('0') << 100 * icrate->fecCrate() + ifec->fecSlot();
      
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {

	  // Add PIA reset description at CCU level
	  index = buildCompleteKey( ifec->fecSlot(), 
				    iring->fecRing(), 
				    iccu->ccuAddr(), 
				    0x30,  // CCU channel
				    0x0 ); // I2C address
	  
	  piaResetDescription* pia = new piaResetDescription( index, 10, 10000, 0xFF );
	  pia->setFecHardwareId( fec_hardware_id.str() );
	  static_pia_resets.push_back( pia );
	  ostringstream os;
	  os << " Added PIA reset at 'CCU level', with address 0x" 
	     << hex << setw(8) << setfill('0') << index << dec;
	  edm::LogInfo(mlConfigDb_) << os;
	  
	}
      }
    }
  }

  if ( static_pia_resets.empty() ) {
    edm::LogError(mlConfigDb_) << "No PIA reset descriptions created!";;
  }
  
  return static_pia_resets;

}
