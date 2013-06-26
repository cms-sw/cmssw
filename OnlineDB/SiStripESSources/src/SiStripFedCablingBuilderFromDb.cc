// Last commit: $Id: SiStripFedCablingBuilderFromDb.cc,v 1.59 2013/05/30 21:52:09 gartung Exp $

#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderFromDb::SiStripFedCablingBuilderFromDb( const edm::ParameterSet& pset ) 
  : SiStripFedCablingESProducer( pset ),
    db_(0),
    source_(sistrip::UNDEFINED_CABLING_SOURCE)
{
  findingRecord<SiStripFedCablingRcd>();
  
  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
  
  // Defined cabling "source" (connections, devices, detids)
  string source = pset.getUntrackedParameter<string>( "CablingSource", "UNDEFINED" );
  source_ = SiStripEnumsAndStrings::cablingSource( source );
  
  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " CablingSource configurable set to \"" << source << "\""
    << ". CablingSource member data set to: \"" 
    << SiStripEnumsAndStrings::cablingSource( source_ ) << "\"";
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderFromDb::~SiStripFedCablingBuilderFromDb() {
  LogTrace(mlCabling_)
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCabling* SiStripFedCablingBuilderFromDb::make( const SiStripFedCablingRcd& ) {
  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Constructing FED cabling...";
   
  // Create FED cabling object 
  SiStripFedCabling* fed_cabling = new SiStripFedCabling();
  
  // Build and retrieve SiStripConfigDb object using service
  db_ = edm::Service<SiStripConfigDb>().operator->(); 

  // Check pointer
  if ( db_ ) {
    edm::LogVerbatim(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Pointer to SiStripConfigDb: 0x" 
      << std::setw(8) << std::setfill('0')
      << std::hex << db_ << std::dec;
  } else {
    edm::LogError(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb returned by DB \"service\"!"
      << " Cannot build FED cabling object!";
    return fed_cabling;
  }
  
  // Check if DB connection is made 
  if ( db_->deviceFactory() || 
       db_->databaseCache() ) { 
    
    // Build FEC cabling object
    SiStripFecCabling fec_cabling;
    buildFecCabling( db_, fec_cabling, source_ );
    
    // Populate FED cabling object
    getFedCabling( fec_cabling, *fed_cabling );
    
    // Call virtual method that writes FED cabling object to conditions DB
    writeFedCablingToCondDb( *fed_cabling );
    
    // Prints FED cabling
    //stringstream ss;
    //ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]" 
    //<< " Printing cabling map..." << endl 
    //<< *fed_cabling;
    //LogTrace(mlCabling_) << ss.str();
    
  } else {
    edm::LogError(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " NULL pointers to DeviceFactory and DatabaseCache returned by SiStripConfigDb!"
      << " Cannot build FED cabling object!";
  }
  
  return fed_cabling;
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripFedCablingBuilderFromDb::buildFecCabling( SiStripConfigDb* const db,
						      SiStripFecCabling& fec_cabling,
						      const sistrip::CablingSource& source ) {

  if      ( source == sistrip::CABLING_FROM_CONNS )       { buildFecCablingFromFedConnections( db, fec_cabling ); }
  else if ( source == sistrip::CABLING_FROM_DEVICES )     { buildFecCablingFromDevices( db, fec_cabling ); }
  else if ( source == sistrip::CABLING_FROM_DETIDS )      { buildFecCablingFromDetIds( db, fec_cabling ); }
  else if ( source == sistrip::UNDEFINED_CABLING_SOURCE ) {
    
    LogTrace(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Unexpected value for CablingSource: \"" 
      << SiStripEnumsAndStrings::cablingSource( source )
      << "\" Querying DB in order to build cabling from one of connections, devices or DetIds...";
    buildFecCabling( db, fec_cabling );
    return;
    
  } else {

    edm::LogError(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Cannot build SiStripFecCabling object!"
      << " sistrip::CablingSource has value: "  
      << SiStripEnumsAndStrings::cablingSource( source );
    return;

  }

  // Debug
  const NumberOfDevices& devs = fec_cabling.countDevices();
  std::stringstream ss;
  ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
     << " Built SiStripFecCabling object with following devices:" 
     << endl << devs;
  edm::LogVerbatim(mlCabling_) << ss.str() << endl;
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripFedCablingBuilderFromDb::buildFecCabling( SiStripConfigDb* const db,
						      SiStripFecCabling& fec_cabling ) {
  LogTrace(mlCabling_)
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Building cabling object...";
  
  if      ( !db->getFedConnections().empty() )     { buildFecCablingFromFedConnections( db, fec_cabling ); }
  else if ( !db->getDeviceDescriptions().empty() ) { buildFecCablingFromDevices( db, fec_cabling ); }
  else if ( !db->getDcuDetIds().empty() )          { buildFecCablingFromDetIds( db, fec_cabling ); }
  else { 
    
    edm::LogError(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Cannot build SiStripFecCabling object!"
      << " FedConnections, DeviceDescriptions and DcuDetIds vectors are all empty!";
    return;

  }
  
  // Debug
  const NumberOfDevices& devices = fec_cabling.countDevices();
  std::stringstream ss;
  ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
     << " Built SiStripFecCabling object with following devices:" 
     << std::endl << devices;
  edm::LogVerbatim(mlCabling_) << ss.str() << endl;
  
}

// -----------------------------------------------------------------------------
/** 
    Populates the SiStripFecCabling conditions object that is
    available via the EventSetup interface. The object contains the
    full FedChannel-Dcu-DetId mapping information.
    
    The map is built using information cached by the SiStripConfigDb
    object, comprising: 1) the FED channel connections, as found in
    the "module.xml" file or database; 2) and Dcu-DetId mapping, as
    found in the "dcuinfo.xml" file or DCU-DetId static table. If any
    information is missing, the method provides "dummy" values.
    
    Methodology:
    
    1) The FEC cabling object is built using FED channel connection
    objects.
    
    2) If the DcuId for a module is null (as defined within the
    connection description), a "dummy" DCU id is provided, based on
    the control key.
    
    3) The cached Dcu-DetId map is queried for a matching DcuId. If
    found, the DetId and ApvPairs are updated. The number of APV pairs
    is checked.
    
    4) If the DCU is not found in the cached map, a "random" DetId is
    assigned (using the remaining "unassigned" DetIds within the
    cached map). The DetId is only assigned if the number of APV pairs
    is consistent with the entry in the cached map.
    
    5) If no appropriate match is found, the DetId is assigned a
    "dummy" value using an incremented counter (starting from 0xFFFF).
    
    6) All Dcu-DetId mappings are accumulated in a new map, and this
    modified map is returned by the method.
*/
void SiStripFedCablingBuilderFromDb::buildFecCablingFromFedConnections( SiStripConfigDb* const db,
									SiStripFecCabling& fec_cabling ) {
  edm::LogVerbatim(mlCabling_)
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Building FEC cabling from FED connections descriptions...";
  
  // ---------- Some initialization ----------
  
  //fec_cabling.clear(); //@@ Need to add method to "clear" FecCabling?
  
  // ---------- Retrieve connection descriptions from database ----------
  
  SiStripConfigDb::FedConnectionsRange conns = db->getFedConnections();
  if ( conns.empty() ) { 
    edm::LogError(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Unable to build FEC cabling!"
      << " No entries in FedConnections vector!";
    return;
  }
  
  // ---------- Retrieve DCU-DetId vector from database ----------

  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Retrieving DCU-DetId vector from database...";
  SiStripConfigDb::DcuDetIdsRange range = db->getDcuDetIds();
  const SiStripConfigDb::DcuDetIdsV dcu_detid_vector( range.begin(), range.end() );
  if ( !dcu_detid_vector.empty() ) { 
    LogTrace(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Found " << dcu_detid_vector.size()
      << " entries in DCU-DetId vector retrieved from database!";
  } else {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No entries in DCU-DetId vector retrieved from database!";
  }

  // ---------- Populate FEC cabling object with retrieved info ----------

  SiStripConfigDb::FedConnectionsV::const_iterator ifed = conns.begin();
  SiStripConfigDb::FedConnectionsV::const_iterator jfed = conns.end();
  for ( ; ifed != jfed; ++ifed ) {
    
    if ( !(*ifed) ) {
      edm::LogWarning(mlCabling_)
	<< "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
	<< " NULL pointer to FedConnection!";
      continue;
    }
    
    //uint16_t fec_id    = static_cast<uint16_t>( (*ifed)->getFecHardwareId() );
    uint16_t fec_crate = static_cast<uint16_t>( (*ifed)->getFecCrateId() + sistrip::FEC_CRATE_OFFSET ); //@@ temporary offset!
    uint16_t fec_slot  = static_cast<uint16_t>( (*ifed)->getFecSlot() );
    uint16_t fec_ring  = static_cast<uint16_t>( (*ifed)->getRingSlot() + sistrip::FEC_RING_OFFSET ); //@@ temporary offset!
    uint16_t ccu_addr  = static_cast<uint16_t>( (*ifed)->getCcuAddress() );
    uint16_t ccu_chan  = static_cast<uint16_t>( (*ifed)->getI2cChannel() );
    uint16_t apv0      = static_cast<uint16_t>( (*ifed)->getApvAddress() );
    uint16_t apv1      = apv0 + 1; //@@ needs implementing!
    uint32_t dcu_id    = static_cast<uint32_t>( (*ifed)->getDcuHardId() );
    uint32_t det_id    = 0; //@@ static_cast<uint32_t>( (*ifed)->getDetId() );
    uint16_t npairs    = 0; //@@ static_cast<uint16_t>( (*ifed)->getApvPairs() ); 
    uint16_t fed_id    = static_cast<uint16_t>( (*ifed)->getFedId() );
    uint16_t fed_ch    = static_cast<uint16_t>( (*ifed)->getFedChannel() );
    uint16_t length    = 0; //@@ static_cast<uint16_t>( (*ifed)->getFiberLength() );

    FedChannelConnection conn( fec_crate, fec_slot, fec_ring, ccu_addr, ccu_chan,
			       apv0, apv1,
			       dcu_id, det_id, npairs,
			       fed_id, fed_ch,
			       length );

    uint16_t fed_crate = sistrip::invalid_; 
    uint16_t fed_slot  = sistrip::invalid_; 
    fed_crate = static_cast<uint16_t>( (*ifed)->getFedCrateId() );
    fed_slot  = static_cast<uint16_t>( (*ifed)->getFedSlot() );
    conn.fedCrate( fed_crate );
    conn.fedSlot( fed_slot );

    fec_cabling.addDevices( conn );

  }
  
  // ---------- Assign DCU and DetIds and then FED cabling ----------
  
  assignDcuAndDetIds( fec_cabling, dcu_detid_vector );
  
}

// -----------------------------------------------------------------------------
/** 
    Builds the SiStripFedCabling conditions object that is available
    via the EventSetup interface. The object contains the full
    FedChannel-Dcu-DetId mapping information.
    
    This method is typically used when the FED connections (ie,
    "module.xml" file) does not exist, such as prior to the FED
    cabling or "bare connection" procedure.

    The map is built using information cached by the SiStripConfigDb
    object, comprising: 1) the hardware device descriptions, as found
    in the "fec.xml" file; 2) and Dcu-DetId mapping, as found in the
    "dcuinfo.xml" file. If any information is missing, the method
    provides "dummy" values.
    
    Methodology:

    The FEC cabling object is built using the hardware device
    descriptions (ie, from "fec.xml"). 

    Given that the FED channel connections are not known, APV pairs
    are cabled to "random" FED ids and channels. FED ids are retrieved
    from any FED descriptions cached by the SiStripConfigDb object
    (ie, from "fed.xml"). A check is made to ensure sufficient FEDs
    exist to cable the entire control system. If not, the shortfall is
    met by generating FED ids using an incremented counter (starting
    from 50).

    If the DcuId (provided by the hardware device descriptions) is
    null, a dummy value is provided, based on the control key.
    
    The Dcu-DetId map (ie, from "dcuinfo.xml") is queried for a
    matching DcuId. If found, the DetId and ApvPairs are updated. If
    not, a "random" DetId within the Dcu-DetId map is assigned. Note
    that a check is made on the number of APV pairs before the DetId
    is assigned. If no appropriate match is found, the DetId is
    assigned a value using an incremented counter (starting from
    0xFFFF).

    All Dcu-DetId mappings are accumulated in a new map, and this
    modified map is returned by the method.
*/
void SiStripFedCablingBuilderFromDb::buildFecCablingFromDevices( SiStripConfigDb* const db,
								 SiStripFecCabling& fec_cabling ) {
  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Building FEC cabling object from device descriptions...";
  
  // ---------- Some initialization ----------

  // fec_cabling.clear(); //@@ Need to add method to "clear" FecCabling?
  
  // ---------- Retrieve APV descriptions from database ----------
  
  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Retrieving APV descriptions from database...";
  SiStripConfigDb::DeviceDescriptionsRange apv_desc = db->getDeviceDescriptions( APV25 );
  if ( !apv_desc.empty() ) { 
    LogTrace(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Retrieved " << apv_desc.size()
      << " APV descriptions from database!";
  } else {
    edm::LogError(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Unable to build FEC cabling!"
      << " No APV descriptions found!";
    return;
  }
  
  // ---------- Retrieve DCU descriptions from database ----------

  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Retrieving DCU descriptions from database...";
  SiStripConfigDb::DeviceDescriptionsRange dcu_desc = db->getDeviceDescriptions( DCU );

  if ( !dcu_desc.empty() ) { 

    uint16_t feh = 0;
    uint16_t ccu = 0;
    SiStripConfigDb::DeviceDescriptionsV::const_iterator idcu;
    for ( idcu = dcu_desc.begin(); idcu != dcu_desc.end(); idcu++ ) {
      dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idcu );
      if ( !dcu ) { continue; }
      if ( dcu->getDcuType() == "FEH" ) { feh++; }
      else { ccu++; }
    }

    LogTrace(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Retrieved " << feh
      << " DCU-FEH descriptions from database!"
      << " (and a further " << ccu << " DCUs for CCU modules, etc...)";

  } else {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No DCU descriptions found!";
  }
  
  // ---------- Retrieve DCU-DetId vector from database ----------


  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Retrieving DCU-DetId vector from database...";
  SiStripConfigDb::DcuDetIdsRange range = db->getDcuDetIds();
  const SiStripConfigDb::DcuDetIdsV dcu_detid_vector( range.begin(), range.end() );
  if ( !dcu_detid_vector.empty() ) { 
    LogTrace(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Found " << dcu_detid_vector.size()
      << " entries in DCU-DetId vector retrieved from database!";
  } else {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No entries in DCU-DetId vector retrieved from database!";
  }

  // ---------- Retrieve FED ids from database ----------
  
  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Retrieving FED ids from database...";
  SiStripConfigDb::FedIdsRange fed_ids = db->getFedIds();
  
  if ( !fed_ids.empty() ) { 
    LogTrace(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Retrieved " << fed_ids.size()
      << " FED ids from database!";
  } else {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No FED ids found!";
  }
  
  // ---------- Populate FEC cabling object with retrieved info ----------

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Building FEC cabling object from APV and DCU descriptions...";
  
  SiStripConfigDb::DeviceDescriptionsV::const_iterator iapv;
  for ( iapv = apv_desc.begin(); iapv != apv_desc.end(); iapv++ ) {
    
    if ( !(*iapv) ) {
      edm::LogWarning(mlCabling_)
	<< "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
	<< " NULL pointer to DeviceDescription (of type APV25)!";
      continue;
    }
    
    SiStripConfigDb::DeviceAddress addr = db->deviceAddress(**iapv);
    FedChannelConnection conn( addr.fecCrate_ + sistrip::FEC_CRATE_OFFSET, //@@ temp
			       addr.fecSlot_, 
			       addr.fecRing_ + sistrip::FEC_RING_OFFSET, //@@ temp
			       addr.ccuAddr_, 
			       addr.ccuChan_, 
			       addr.i2cAddr_ ); 
    fec_cabling.addDevices( conn );

  } 
  
  SiStripConfigDb::DeviceDescriptionsV::const_iterator idcu;
  for ( idcu = dcu_desc.begin(); idcu != dcu_desc.end(); idcu++ ) {

    if ( !(*idcu) ) {
      edm::LogWarning(mlCabling_)
	<< "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
	<< " NULL pointer to DeviceDescription (of type DCU)!";
      continue;
    }

    SiStripConfigDb::DeviceAddress addr = db->deviceAddress(**idcu);
    dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idcu );
    if ( !dcu ) { continue; }
    if ( dcu->getDcuType() != "FEH" ) { continue; }
    FedChannelConnection conn( addr.fecCrate_ + sistrip::FEC_CRATE_OFFSET, //@@ temp,
			       addr.fecSlot_, 
			       addr.fecRing_ + sistrip::FEC_RING_OFFSET, //@@ temp, 
			       addr.ccuAddr_, 
			       addr.ccuChan_,
			       0, 0, // APV I2C addresses not used
			       dcu->getDcuHardId() ); 
    fec_cabling.dcuId( conn );

  }

  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Finished building FEC cabling object from APV and DCU descriptions!";

  NumberOfDevices devs1 = fec_cabling.countDevices();
  std::stringstream ss1;
  ss1 << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Number of devices in FEC cabling object:" << std::endl;
  devs1.print(ss1);
  LogTrace(mlCabling_) << ss1.str();

  // ---------- Counters used in assigning "dummy" FED ids and channels ----------
  
  std::vector<uint16_t>::const_iterator ifed = fed_ids.begin();
  uint16_t fed_ch = 0;
  
  // ---------- Assign "dummy" FED crates/slots/ids/chans to constructed modules ----------

  std::vector<uint32_t> used_keys;

  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Randomly assigning FED ids/channels to APV pairs in front-end modules...";

  if ( fed_ids.empty() ) {
    edm::LogError(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No FED ids retrieved from database! Unable to cable system!";
  } else {

    bool complete = false;
    std::vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin();
    std::vector<SiStripFecCrate>::const_iterator jcrate = fec_cabling.crates().end();
    while ( !complete && icrate != jcrate ) {
      std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin();
      std::vector<SiStripFec>::const_iterator jfec = icrate->fecs().end();
      while ( !complete && ifec != jfec ) {
	std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin();
	std::vector<SiStripRing>::const_iterator jring = ifec->rings().end();
	while ( !complete && iring != jring ) {
	  std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); 
	  std::vector<SiStripCcu>::const_iterator jccu = iring->ccus().end(); 
	  while ( !complete && iccu != jccu ) {
	    std::vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); 
	    std::vector<SiStripModule>::const_iterator jmod = iccu->modules().end(); 
	    while ( !complete && imod != jmod ) {
	    
	      // Set number of APV pairs based on devices found 
	      const_cast<SiStripModule&>(*imod).nApvPairs(0); 
	      
	      used_keys.push_back( SiStripFecKey( imod->fecCrate(),
						  imod->fecSlot(), 
						  imod->fecRing(),
						  imod->ccuAddr(), 
						  imod->ccuChan() ).key() );
	      
// 	      // Add middle LLD channel if missing (to guarantee all FED channels are cabled!)
// 	      if ( imod->nApvPairs() == 2 ) {
// 		const_cast<SiStripModule&>(*imod).nApvPairs(3); 
// 		FedChannelConnection temp( imod->fecCrate(),
// 					   imod->fecSlot(), 
// 					   imod->fecRing(),
// 					   imod->ccuAddr(), 
// 					   imod->ccuChan(), 
// 					   SiStripFecKey::i2cAddr(2,true),
// 					   SiStripFecKey::i2cAddr(2,false) ); 
// 		const_cast<SiStripModule&>(*imod).addDevices( temp );
// 	      }
// 	      const_cast<SiStripModule&>(*imod).nApvPairs(0); 
	    
	      // Iterate through APV pairs 
	      for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {
	      
		// Check FED id and channel
		if ( ifed == fed_ids.end() ) { fed_ch++; ifed = fed_ids.begin(); } 
		if ( fed_ch == 96 ) {
		  edm::LogWarning(mlCabling_)
		    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		    << " Insufficient FED channels to cable all devices in control system!";
		  complete = true;
		  break;
		}
	      
		// Set "dummy" FED id and channel
		pair<uint16_t,uint16_t> addr = imod->activeApvPair( imod->lldChannel(ipair) );
		SiStripModule::FedChannel fed_channel( (*ifed)/16+1, // 16 FEDs per crate, numbering starts from 1
						       (*ifed)%16+2, // FED slot starts from 2
						       *ifed, 
						       fed_ch );
		const_cast<SiStripModule&>(*imod).fedCh( addr.first, fed_channel );
		ifed++;
	      
	      }
	    
	      imod++;
	    }
	    iccu++;
	  }
	  iring++;
	}
	ifec++;
      }
      icrate++;
    }

  }

  std::sort( used_keys.begin(), used_keys.end() );

  NumberOfDevices devs2 = fec_cabling.countDevices();
  std::stringstream ss2;
  ss2 << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Number of devices in FEC cabling object:" << std::endl;
  devs2.print(ss2);
  LogTrace(mlCabling_) << ss2.str();

  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Finished randomly assigning FED ids/channels to APV pairs in front-end modules...";
 
  // ---------- Assign "dummy" devices to remaining FED ids/chans ----------

  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Assigning APV pairs in dummy front-end modules to any remaining \"uncabled\" FED ids/channels...";

  if ( fed_ids.empty() ) {
    edm::LogError(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No FED ids retrieved from database! Unable to cable system!";
  } else {

    uint16_t module = 0;
    bool complete = false;
    while ( !complete ) { 
      for ( uint16_t lld = sistrip::LLD_CHAN_MIN; lld < sistrip::LLD_CHAN_MAX+1; lld++ ) {
      
	// Check FED id and channel
	if ( ifed == fed_ids.end() ) { fed_ch++; ifed = fed_ids.begin(); } 
	if ( fed_ch == 96 ) {
	  LogTrace(mlCabling_)
	    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
	    << " All FED channels are now cabled!";
	  complete = true;
	  break;
	}

	// commented because key is not used
	//uint32_t key = SiStripFecKey( fecCrate( module ), 
	//			      fecSlot( module ), 
	//			      fecRing( module ), 
	//			      ccuAddr( module ), 
	//			      ccuChan( module ) ).key();
	
	//if ( std::find( used_keys.begin(), used_keys.end(), key ) != used_keys.end() ) { break; }
	
	FedChannelConnection temp( fecCrate( module ), 
				   fecSlot( module ), 
				   fecRing( module ), 
				   ccuAddr( module ), 
				   ccuChan( module ), 
				   SiStripFecKey::i2cAddr(lld,true),
				   SiStripFecKey::i2cAddr(lld,false),
				   sistrip::invalid32_,
				   sistrip::invalid32_,
				   3, // npairs
				   *ifed, 
				   fed_ch );
	uint16_t fed_crate = (*ifed)/16+1; // 16 FEDs per crate, numbering starts from 1
	uint16_t fed_slot  = (*ifed)%16+2; // FED slot starts from 2
	temp.fedCrate( fed_crate );
	temp.fedSlot( fed_slot );
	fec_cabling.addDevices( temp );
	ifed++;
      
      } 
      module++;
    }

  }

  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Finished assigning APV pairs in dummy front-end modules to any remaining \"uncabled\" FED ids/channels...";

  NumberOfDevices devs3 = fec_cabling.countDevices();
  std::stringstream ss3;
  ss3 << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Number of devices in FEC cabling object:" << std::endl;
  devs3.print(ss3);
  LogTrace(mlCabling_) << ss3.str();

  // ---------- Assign DCU and DetIds and then FED cabling ----------
  
  assignDcuAndDetIds( fec_cabling, dcu_detid_vector );
  
}

// -----------------------------------------------------------------------------
/**
   Builds the SiStripFedCabling conditions object that is available
   via the EventSetup interface. The object contains the full
   FedChannel-Dcu-DetId mapping information.
    
   This method is typically used when only the Dcu-DetId map (ie,
   from "dcuinfo.xml") exists and the FED connections (ie,
   "module.xml" file) and device descriptions (ie, from "fec.xml")
   are both missing.
    
   The map is built using the Dcu-DetId map that is cached by the
   SiStripConfigDb object. As a minimum, the map should contain
   values within both the DetId and APpvPair fields, but if any
   information is missing, the method provides "dummy" values.
    
   Methodology:

   The FEC cabling object is built using the Dcu-DetId map (ie,
   from "dcuinfo.xml"). For each entry, the DcuId, DetId and ApvPairs
   values are retrieved. For each ApvPair, a FED channel connection
   object is created using "dummy" hardware addresses.

   If the DcuId (provided by the hardware device descriptions) is
   null, a dummy value is provided, based on the control key.
    
   If the DetId is null, a value is assigned using an incremented
   counter (starting from 0xFFFF). 

   If the number of APV pairs is null, a value of 2 or 3 is randomly
   assigned.

   Given that the FED channel connections are not known, APV pairs
   are cabled to "random" FED ids and channels. 

   All Dcu-DetId mappings are accumulated in a new map, and this
   modified map is returned by the method.
*/
void SiStripFedCablingBuilderFromDb::buildFecCablingFromDetIds( SiStripConfigDb* const db,
								SiStripFecCabling& fec_cabling ) {
  edm::LogVerbatim(mlCabling_)
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Building FEC cabling object from DetIds...";
  
  // ---------- Some initialization ----------

  // fec_cabling.clear();
  
  // chans_per_ring = chans_per_ccu * ccus_per_ring = 100 (TOB gives lower limit of 60)
  uint32_t chans_per_ccu  = 10; 
  uint32_t ccus_per_ring  = 10;
  uint32_t rings_per_fec  = 8;
  uint32_t fecs_per_crate = 11;

  // ---------- Retrieve necessary descriptions from database ----------


  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Retrieving DCU-DetId vector from database...";
  SiStripConfigDb::DcuDetIdsRange range = db->getDcuDetIds();
  const SiStripConfigDb::DcuDetIdsV dcu_detid_vector( range.begin(), range.end() );
  if ( !dcu_detid_vector.empty() ) { 
    LogTrace(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Found " << dcu_detid_vector.size()
      << " entries in DCU-DetId vector retrieved from database!";
  } else {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No entries in DCU-DetId vector retrieved from database!"
      << " Unable to build FEC cabling!";
    return;
  }

  // ---------- Populate FEC cabling object with DCU, DetId and "dummy" control info ----------

  uint32_t imodule = 0;
  SiStripConfigDb::DcuDetIdsV::const_iterator iter;
  for ( iter = dcu_detid_vector.begin(); iter != dcu_detid_vector.end(); iter++ ) {

    if ( !(iter->second) ) {
      edm::LogWarning(mlCabling_)
	<< "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
	<< " NULL pointer to TkDcuInfo!";
      continue;
    }
    
    uint16_t fec_crate = ( imodule / ( chans_per_ccu * ccus_per_ring * rings_per_fec * fecs_per_crate ) ) + 1;
    uint16_t fec_slot  = ( imodule / ( chans_per_ccu * ccus_per_ring * rings_per_fec ) ) % fecs_per_crate + 2;
    uint16_t fec_ring  = ( imodule / ( chans_per_ccu * ccus_per_ring ) ) % rings_per_fec + 1;
    uint16_t ccu_addr  = ( imodule / ( chans_per_ccu) ) % ccus_per_ring + 1;
    uint16_t ccu_chan  = ( imodule ) % chans_per_ccu + 16;
    
    uint32_t dcu_id = iter->second->getDcuHardId(); // 
    uint32_t det_id = iter->second->getDetId();
    uint16_t npairs = iter->second->getApvNumber()/2;
    uint16_t length = (uint16_t) iter->second->getFibreLength(); //@@ should be double!

    // --- Check if DCU, DetId and nApvPairs are null ---

    if ( !dcu_id ) {
      dcu_id = SiStripFecKey( fec_crate,
			      fec_slot,
			      fec_ring,
			      ccu_addr,
			      ccu_chan ).key();
    }
    if ( !det_id ) { det_id = 0xFFFF + imodule; } 
    if ( !npairs ) { npairs = rand()/2 ? 2 : 3; }
    
    // --- Construct FedChannelConnection objects ---

    for ( uint16_t ipair = 0; ipair < npairs; ipair++ ) {
      uint16_t iapv = ( ipair == 1 && npairs == 2 ? 36 : 32 + 2 * ipair ) ;
      FedChannelConnection conn( fec_crate, 
				 fec_slot, 
				 fec_ring, 
				 ccu_addr, 
				 ccu_chan, 
				 iapv, iapv+1,
				 dcu_id, det_id, npairs,
				 0, 0, // FED id and channel
				 length,
				 true, true, true, true );
      fec_cabling.addDevices( conn );
    } 

    imodule++;
  }

  // ---------- Assign "dummy" FED ids/chans to Modules of FEC cabling object ----------
  
  uint32_t fed_id = 50;
  uint32_t fed_ch = 0;
  for ( vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin(); icrate != fec_cabling.crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    if ( 96-fed_ch < imod->nApvPairs() ) { fed_id++; fed_ch = 0; } // move to next FED
	    for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {
	      pair<uint16_t,uint16_t> addr = imod->activeApvPair( (*imod).lldChannel(ipair) );
	      SiStripModule::FedChannel fed_channel( (fed_id-50)/16+1, // 16 FEDs per crate, numbering starts from 1
						     (fed_id-50)%16+2, // FED slot starts from 2
						     fed_id, 
						     fed_ch );
	      const_cast<SiStripModule&>(*imod).fedCh( addr.first, fed_channel );
	      fed_ch++;
	    }
	  }
	}
      }
    }
  }

}

// -----------------------------------------------------------------------------
/** */
void SiStripFedCablingBuilderFromDb::assignDcuAndDetIds( SiStripFecCabling& fec_cabling,
							 const std::vector< std::pair<uint32_t,TkDcuInfo*> >& _in ) {
  std::vector< std::pair<uint32_t,TkDcuInfo*> > in = _in; 
  // ---------- Check if entries found in DCU-DetId vector ----------

  if ( in.empty() ) { 
    edm::LogError(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No entries in DCU-DetId vector!";
  }

  // ---------- Assign DCU and DetId to Modules in FEC cabling object ----------

  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Assigning DCU ids and DetIds to constructed modules...";

  uint16_t channels = 0;
  uint16_t six      = 0;
  uint16_t four     = 0;
  uint16_t unknown  = 0;
  uint16_t missing  = 0;
  
  for ( vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin(); icrate != fec_cabling.crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    SiStripModule& module = const_cast<SiStripModule&>(*imod);

	    //@@ TEMP FIX UNTIL LAURENT DEBUGS FedChannelConnectionDescription CLASS
	    module.nApvPairs(0);
	    
	    // --- Check for null DCU ---
	    
	    if ( !module.dcuId() ) { 
	      SiStripFecKey path( icrate->fecCrate(),
				  ifec->fecSlot(), 
				  iring->fecRing(), 
				  iccu->ccuAddr(), 
				  imod->ccuChan() );
	      uint32_t module_key = path.key();
	      module.dcuId( module_key ); // Assign DCU id equal to control key
	      stringstream ss;
	      ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		 << " Found NULL DcuId! Setting 'dummy' value based control key 0x"
		 << hex << setw(8) << setfill('0') << module_key << dec;
	      edm::LogWarning(mlCabling_) << ss.str();
	    }
	    
	    // --- Check for null DetId ---
	    
	    if ( !module.detId() ) { 
	      
	      // --- Search for DcuId in map ---
	      
	      SiStripConfigDb::DcuDetIdsV::iterator iter = in.end();
	      iter = SiStripConfigDb::findDcuDetId( in.begin(), in.end(), module.dcuId() );
	      if ( iter != in.end() ) { 
		
		if ( !(iter->second) ) {
		  edm::LogWarning(mlCabling_)
		    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		    << " NULL pointer to TkDcuInfo!";
		  continue;
		}

		// --- Assign DetId and set nApvPairs based on APVs found in given Module ---
		
		module.detId( iter->second->getDetId() ); 
		module.nApvPairs(0); 
		
		// count expected channels
		uint16_t pairs = iter->second->getApvNumber()/2;
		channels += pairs;
		if ( pairs == 2 ) { four++; }
		else if ( pairs == 3 ) { six++; }
		else { unknown++; }
		
		// --- Check number of APV pairs is valid and consistent with cached map ---

		if ( module.nApvPairs() != 2 && module.nApvPairs() != 3 ) { 

		  missing += ( iter->second->getApvNumber()/2 - module.nApvPairs() );
		  stringstream ss1;
		  ss1 << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]" << std::endl
		      << " Module with DCU id 0x" 
		      << hex << setw(8) << setfill('0') << module.dcuId() << dec
		      << " and DetId 0x"
		      << hex << setw(8) << setfill('0') << module.detId() << dec
		      << " has unexpected number of APV pairs (" 
		      << module.nApvPairs() << ")." << std::endl
		      << " Some APV pairs may have not been detected by the FEC scan." << std::endl
		      << " Setting to value found in static map (" 
		      << iter->second->getApvNumber()/2 << ")...";
		  edm::LogWarning(mlCabling_) << ss1.str();
		  module.nApvPairs( iter->second->getApvNumber()/2 ); 

		} else if ( module.nApvPairs() < iter->second->getApvNumber()/2 ) {

		  missing += ( iter->second->getApvNumber()/2 - module.nApvPairs() );
		  stringstream ss2;
		  ss2 << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]" << std::endl
		      << " Module with DCU id 0x" 
		      << hex << setw(8) << setfill('0') << module.dcuId() << dec
		      << " and DetId 0x"
		      << hex << setw(8) << setfill('0') << module.detId() << dec
		      << " has number of APV pairs (" 
		      << module.nApvPairs() 
		      << ") that does not match value found in DCU-DetId vector (" 
		      << iter->second->getApvNumber()/2 << ")." << std::endl
		      << " Some APV pairs may have not been detected by"
		      << " the FEC scan or the DCU-DetId vector may be incorrect." << std::endl
		      << " Setting to value found in static map ("
		      << iter->second->getApvNumber()/2 << ")...";
		  edm::LogWarning(mlCabling_) << ss2.str();
		  module.nApvPairs( iter->second->getApvNumber()/2 ); 

		}
		
		// --- Check for null fibre length ---
		
		if ( !module.length() ) { 
		  module.length( static_cast<uint16_t>( iter->second->getFibreLength() ) );
		}

		// --- Remove TkDcuInfo object from cached map ---

		in.erase( iter );

	      } // Set for DCU in static table
	    } // Check for null DetId

	  } // Module loop
	} // CCU loop
      } // FEC ring loop
    } // FEC loop
  } // FEC crate loop

  std::stringstream sss;
  sss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]" << std::endl
      << " Connections in DCU-DetId map : " << channels << std::endl
      << " 4-APV modules                : " << four << std::endl
      << " 6-APV modules                : " << six << std::endl
      << " Unknown number of APV pairs  : " << unknown << std::endl
      << " Total found APV pairs        : " << ( channels - missing ) << std::endl
      << " Total missing APV pairs      : " << missing << std::endl;
  edm::LogVerbatim(mlCabling_) << sss.str();
  
  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Finished assigning DCU ids and DetIds to constructed modules...";

  // ---------- "Randomly" assign DetIds to Modules with DCU ids not found in static table ----------

  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Assigning \"random\" DetIds to modules with DCU ids not found in static table...";
  
  uint32_t detid = 0x10000; // Incremented "dummy" DetId
  for ( vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin(); icrate != fec_cabling.crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    SiStripModule& module = const_cast<SiStripModule&>(*imod);
	    
	    // --- Check for null DetId and search for DCU in cached map ---
	    
	    if ( !module.detId() ) { 
	      
	      SiStripConfigDb::DcuDetIdsV::iterator iter = in.end();
	      iter = SiStripConfigDb::findDcuDetId( in.begin(), in.end(), module.dcuId() );
	      if ( iter != in.end() ) { 
				
		// --- Search for "random" module with consistent number of APV pairs ---
		
		SiStripConfigDb::DcuDetIdsV::iterator idcu;
		if ( in.empty() ) { idcu = in.end(); }
		else {
		  idcu = in.begin();
		  while ( idcu != in.end() ) { 
		    if ( idcu->second ) {
		      if ( static_cast<uint32_t>(idcu->second->getApvNumber()) == 
			   static_cast<uint32_t>(2*module.nApvPairs()) ) { break; }
		    }
		    idcu++; 
		  }
		} 
		
		// --- Assign "random" DetId if number of APV pairs is consistent ---
		
		if ( idcu != in.end() ) {
		  
		  if ( !(idcu->second) ) {
		    edm::LogWarning(mlCabling_)
		      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		      << " NULL pointer to TkDcuInfo!";
		    continue;
		  }

		  module.detId( idcu->second->getDetId() );
		  in.erase( idcu );
		  
		  stringstream ss;
		  ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		     << " Did not find module with DCU id 0x"
		     << hex << setw(8) << setfill('0') << module.dcuId() << dec
		     << " in DCU-DetId vector!" << endl
		     << " Assigned 'random' DetId 0x"
		     << hex << setw(8) << setfill('0') << module.detId() << dec;
		  edm::LogWarning(mlCabling_) << ss.str();
		  
		} else { // --- Else, assign "dummy" DetId based on counter ---
		  
		  // If no match found, then assign DetId using incremented counter
		  module.detId( detid );
		  detid++;

		  stringstream ss;
		  if ( in.empty() ) {
		    ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		       << " Did not find module with DCU id 0x"
		       << hex << setw(8) << setfill('0') << module.dcuId() << dec
		       << " in DCU-DetId vector!" 
		       << " Could not assign 'random' DetId as DCU-DetID map is empty!"
		       << " Assigned DetId based on incremented counter, with value 0x"
		       << hex << setw(8) << setfill('0') << module.detId() << dec;
		  } else {
		    ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		       << " Did not find module with DCU id 0x"
		       << hex << setw(8) << setfill('0') << module.dcuId() << dec
		       << " in DCU-DetId vector!" 
		       << " Could not assign 'random' DetId as no modules had appropriate number of APV pairs ("
		       << module.nApvPairs()
		       << "). Assigned DetId based on incremented counter, with value 0x"
		       << hex << setw(8) << setfill('0') << module.detId() << dec;
		  }
		  edm::LogWarning(mlCabling_) << ss.str();

		}
	      }

	    } 

	  } // Module loop
	} // CCU loop
      } // FEC ring loop
    } // FEC loop
  } // FEC crate loop

  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Finished assigning \"random\" DetIds to modules with DCU ids not found in static table...";
  
  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Assigning \"random\" DetIds to modules with DCU ids not found in static table...";
  
  // ---------- Check for unassigned DetIds ----------
  
  if ( !in.empty() ) {
    stringstream ss;
    ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
       << " Not all DetIds have been assigned to a DcuId! " 
       << in.size() << " DetIds are unassigned!";
    edm::LogWarning(mlCabling_) << ss.str();
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripFedCablingBuilderFromDb::getFedCabling( const SiStripFecCabling& fec_cabling, 
						    SiStripFedCabling& fed_cabling ) {
  vector<FedChannelConnection> conns;
  fec_cabling.connections( conns );
  fed_cabling.buildFedCabling( conns );
}

// -----------------------------------------------------------------------------
/** */
void SiStripFedCablingBuilderFromDb::getFecCabling( const SiStripFedCabling& fed_cabling, 
						    SiStripFecCabling& fec_cabling ) {
  fec_cabling.buildFecCabling( fed_cabling );
}

// -----------------------------------------------------------------------------
// 
void SiStripFedCablingBuilderFromDb::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& key, 
						     const edm::IOVSyncValue& iov_sync, 
						     edm::ValidityInterval& iov_validity ) {
  edm::ValidityInterval infinity( iov_sync.beginOfTime(), iov_sync.endOfTime() );
  iov_validity = infinity;
}

