// Last commit: $Id: SiStripFedCablingBuilderFromDb.cc,v 1.36 2007/06/04 12:56:44 bainbrid Exp $

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
time_t SiStripFedCablingBuilderFromDb::timer_ = time(NULL);

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderFromDb::SiStripFedCablingBuilderFromDb( const edm::ParameterSet& pset ) 
  : SiStripFedCablingESSource( pset ),
    db_(0),
    source_(sistrip::UNDEFINED_CABLING_SOURCE)
{
  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
  
  // Defined cabling "source" (connections, devices, detids)
  string source = pset.getUntrackedParameter<string>( "CablingSource", "UNDEFINED" );
  source_ = SiStripEnumsAndStrings::cablingSource( source );

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " CablingSource configurable set to \"" << source << "\""
    << ". CablingSource member data set to: \"" 
    << SiStripEnumsAndStrings::cablingSource( source_ ) << "\"";
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderFromDb::~SiStripFedCablingBuilderFromDb() {
  edm::LogVerbatim(mlCabling_)
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCabling* SiStripFedCablingBuilderFromDb::makeFedCabling() {
  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Constructing FED cabling...";
   
  // Create FED cabling object 
  SiStripFedCabling* fed_cabling = new SiStripFedCabling();
  
  // Build and retrieve SiStripConfigDb object using service
  db_ = edm::Service<SiStripConfigDb>().operator->(); //@@ NOT GUARANTEED TO BE THREAD SAFE! 
  edm::LogWarning(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Nota bene: using the SiStripConfigDb API"
    << " as a \"service\" does not presently guarantee"
    << " thread-safe behaviour!...";
  
  // Check if DB connection is made 
  if ( db_ ) { 

    if ( db_->deviceFactory() ) { 
      
      // Build FEC cabling object
      SiStripFecCabling fec_cabling;
      SiStripConfigDb::DcuDetIdMap dcu_detid_map;
      buildFecCabling( db_, fec_cabling, dcu_detid_map, source_ );
      
      // Populate FED cabling object
      getFedCabling( fec_cabling, *fed_cabling );
      
      // Call virtual method that writes FED cabling object to conditions DB
      writeFedCablingToCondDb( *fed_cabling );
      
      // Prints FED cabling
      stringstream ss;
      ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]" 
	 << " Printing cabling map..." << endl 
	 << *fed_cabling;
      LogTrace(mlCabling_) << ss.str();
      
    } else {
      edm::LogWarning(mlCabling_)
	<< "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
	<< " NULL pointer to DeviceFactory returned by SiStripConfigDb!"
	<< " Cannot build FED cabling object!";
    }
  } else {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb returned by DB \"service\"!"
      << " Cannot build FED cabling object!";
  }
  
  return fed_cabling;
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripFedCablingBuilderFromDb::buildFecCabling( SiStripConfigDb* const db,
						      SiStripFecCabling& fec_cabling,
						      SiStripConfigDb::DcuDetIdMap& new_map,
						      const sistrip::CablingSource& source ) {

  if ( source == sistrip::CABLING_FROM_CONNS ) {
    
    buildFecCablingFromFedConnections( db, fec_cabling, new_map ); 
    
  } else if ( source == sistrip::CABLING_FROM_DEVICES ) {
    
    buildFecCablingFromDevices( db, fec_cabling, new_map ); 
    
  } else if ( source == sistrip::CABLING_FROM_DETIDS ) {
    
    buildFecCablingFromDetIds( db, fec_cabling, new_map ); 
    
  } else if ( source == sistrip::UNDEFINED_CABLING_SOURCE ) {
    
    edm::LogVerbatim(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Unexpected value for CablingSource: \"" 
      << SiStripEnumsAndStrings::cablingSource( source )
      << "\" Querying DB in order to build cabling from one of connections, devices or DetIds...";
    buildFecCabling( db, fec_cabling, new_map );
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
						      SiStripFecCabling& fec_cabling,
						      SiStripConfigDb::DcuDetIdMap& new_map ) {
  edm::LogVerbatim(mlCabling_)
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Checking contents of database (this may take some time)...";
  
  if ( !db->getFedConnections().empty() ) {
    buildFecCablingFromFedConnections( db, fec_cabling, new_map ); 
  } else if ( !db->getDeviceDescriptions().empty() ) {
    buildFecCablingFromDevices( db, fec_cabling, new_map ); 
  } else if ( !db->getDcuDetIdMap().empty() ) {
    buildFecCablingFromDetIds( db, fec_cabling, new_map ); 
  } else { 
    edm::LogError(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Cannot build SiStripFecCabling object!"
      << " FedConnections, DeviceDescriptions and DcuDetIdMap vectors are all empty!";
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
									SiStripFecCabling& fec_cabling,
									SiStripConfigDb::DcuDetIdMap& used_map ) {
  edm::LogVerbatim(mlCabling_)
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Building FEC cabling from FED connections descriptions...";
  
  // ---------- Some initialization ----------
  
  used_map.clear(); 
  //fec_cabling.clear(); //@@ Need to add method to "clear" FecCabling?
  
  // ---------- Retrieve necessary descriptions from database ----------
  
  const SiStripConfigDb::FedConnections& conns = db->getFedConnections();
  if ( conns.empty() ) { 
    edm::LogWarning(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Unable to build FEC cabling!"
      << " No entries in FedConnections vector!";
    return;
  }
  
  SiStripConfigDb::DcuDetIdMap cached_map = db->getDcuDetIdMap();
  
  // ---------- Populate FEC cabling object with retrieved info ----------

  SiStripConfigDb::FedConnections::const_iterator ifed = conns.begin();
  for ( ; ifed != conns.end(); ifed++ ) {
    uint16_t fec_crate = static_cast<uint16_t>( (*ifed)->getFecInstance() + sistrip::FEC_CRATE_OFFSET ); //@@ temporary offset!
    uint16_t fec_slot  = static_cast<uint16_t>( (*ifed)->getSlot() );
    uint16_t fec_ring  = static_cast<uint16_t>( (*ifed)->getRing() + sistrip::FEC_RING_OFFSET ); //@@ temporary offset!
    uint16_t ccu_addr  = static_cast<uint16_t>( (*ifed)->getCcu() );
    uint16_t ccu_chan  = static_cast<uint16_t>( (*ifed)->getI2c() );
    uint16_t apv0      = static_cast<uint16_t>( (*ifed)->getApv() );
    uint16_t apv1      = apv0 + 1; //@@ needs implementing!
    uint32_t dcu_id    = static_cast<uint32_t>( (*ifed)->getDcuHardId() );
    uint32_t det_id    = static_cast<uint32_t>( (*ifed)->getDetId() );
    uint16_t npairs    = static_cast<uint16_t>( (*ifed)->getApvPairs() );
    uint16_t fed_id    = static_cast<uint16_t>( (*ifed)->getFedId() );
    uint16_t fed_ch    = static_cast<uint16_t>( (*ifed)->getFedChannel() ); // "internal" numbering scheme
    FedChannelConnection conn( fec_crate, fec_slot, fec_ring, ccu_addr, ccu_chan,
			       apv0, apv1,
			       dcu_id, det_id, npairs,
			       fed_id, fed_ch );
    fec_cabling.addDevices( conn );
  }
  
  // ---------- Assign DCU and DetIds and then FED cabling ----------
  
  assignDcuAndDetIds( fec_cabling, cached_map, used_map );
  
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
								 SiStripFecCabling& fec_cabling,
								 SiStripConfigDb::DcuDetIdMap& used_map ) {
  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Building FEC cabling object from device descriptions...";
  
  // ---------- Some initialization ----------

  used_map.clear();
  // fec_cabling.clear();
  SiStripFedCablingBuilderFromDb::timer_ = time(NULL);
  
  // ---------- Retrieve APV descriptions from database ----------
  
  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Retrieving APV descriptions from database...";
  SiStripConfigDb::DeviceDescriptions apv_desc;
  db->getDeviceDescriptions( apv_desc, APV25 );
  if ( !apv_desc.empty() ) { 
    edm::LogVerbatim(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Retrieved " << apv_desc.size()
      << " APV descriptions from database!";
  } else {
    edm::LogWarning(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Unable to build FEC cabling!"
      << " No APV descriptions found!";
    return;
  }

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Cumulative time [s]: " 
    << time(NULL) - SiStripFedCablingBuilderFromDb::timer_;
  
  // ---------- Retrieve DCU descriptions from database ----------

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Retrieving DCU descriptions from database...";
  SiStripConfigDb::DeviceDescriptions dcu_desc;
  db->getDeviceDescriptions( dcu_desc, DCU );
  if ( !dcu_desc.empty() ) { 
    uint16_t feh = 0;
    uint16_t ccu = 0;
    SiStripConfigDb::DeviceDescriptions::const_iterator idcu;
    for ( idcu = dcu_desc.begin(); idcu != dcu_desc.end(); idcu++ ) {
      dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idcu );
      if ( !dcu ) { continue; }
      if ( dcu->getDcuType() == "FEH" ) { feh++; }
      else { ccu++; }
    }
    edm::LogVerbatim(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Retrieved " << feh
      << " DCU-FEH descriptions from database!"
      << " (and a further " << ccu << " DCUs for CCU modules, etc...)";
  } else {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No DCU descriptions found!";
  }

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Cumulative time [s]: " 
    << time(NULL) - SiStripFedCablingBuilderFromDb::timer_;
  
  // ---------- Retrieve DCU-DetId map from database ----------

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Retrieving DCU-DetId map from database...";
  SiStripConfigDb::DcuDetIdMap cached_map = db->getDcuDetIdMap();
  if ( !cached_map.empty() ) { 
    edm::LogVerbatim(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Found " << cached_map.size()
      << " entries in DCU-DetId map retrieved from database!";
  } else {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No entries in DCU-DetId map retrieved from database!";
  }

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Cumulative time [s]: " 
    << time(NULL) - SiStripFedCablingBuilderFromDb::timer_;

  // ---------- Retrieve FED ids from database ----------
  
  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Retrieving FED ids from database...";
  vector<uint16_t> fed_ids = db->getFedIds();
  if ( !fed_ids.empty() ) { 
    edm::LogVerbatim(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Retrieved " << fed_ids.size()
      << " FED ids from database!";
  } else {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No FED ids found!";
  }

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Cumulative time [s]: " 
    << time(NULL) - SiStripFedCablingBuilderFromDb::timer_;
  
  // ---------- Populate FEC cabling object with retrieved info ----------

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Building FEC cabling object from APV and DCU descriptions...";
  
  SiStripConfigDb::DeviceDescriptions::const_iterator iapv;
  for ( iapv = apv_desc.begin(); iapv != apv_desc.end(); iapv++ ) {
    const SiStripConfigDb::DeviceAddress& addr = db->deviceAddress(**iapv);
    FedChannelConnection conn( addr.fecCrate_ + sistrip::FEC_CRATE_OFFSET, //@@ temp
			       addr.fecSlot_, 
			       addr.fecRing_ + sistrip::FEC_RING_OFFSET, //@@ temp
			       addr.ccuAddr_, 
			       addr.ccuChan_, 
			       addr.i2cAddr_ ); 
    fec_cabling.addDevices( conn );
  } 
  
  SiStripConfigDb::DeviceDescriptions::const_iterator idcu;
  for ( idcu = dcu_desc.begin(); idcu != dcu_desc.end(); idcu++ ) {
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

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Finished building FEC cabling object from APV and DCU descriptions!";

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Cumulative time [s]: "
    << time(NULL) - SiStripFedCablingBuilderFromDb::timer_;

  // ---------- Counters used in assigning "dummy" FED ids and channels ----------

  vector<uint16_t>::iterator ifed = fed_ids.begin();
  uint16_t fed_ch = 0;

  // ---------- Assign "dummy" FED ids/chans to constructed modules ----------

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Assigning \"dummy\" FED ids/channels to constructed modules...";

  for ( vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin(); icrate != fec_cabling.crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    
	    // Set number of APV pairs based on devices found 
	    const_cast<SiStripModule&>(*imod).nApvPairs(0); 
	    
	    // Add middle LLD channel if missing
	    if ( imod->nApvPairs() == 2 ) {
	      const_cast<SiStripModule&>(*imod).nApvPairs(3); 
	      FedChannelConnection temp( imod->fecCrate(),
					 imod->fecSlot(), 
					 imod->fecRing(),
					 imod->ccuAddr(), 
					 imod->ccuChan(), 
					 SiStripFecKey::i2cAddr(2,true),
					 SiStripFecKey::i2cAddr(2,false) ); 
	      const_cast<SiStripModule&>(*imod).addDevices( temp );
	    }
	    const_cast<SiStripModule&>(*imod).nApvPairs(0); 
	    
	    // Iterate through APV pairs 
	    for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {
	      
	      // Check FED id and channel
	      if ( ifed == fed_ids.end() ) { fed_ch++; ifed = fed_ids.begin(); } 
	      if ( fed_ch == 96 ) {
		edm::LogWarning(mlCabling_)
		  << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		  << " Insufficient FED channels to cable entire system!";
		break;
	      }
	      
	      // Set "dummy" FED id and channel
	      pair<uint16_t,uint16_t> addr = imod->activeApvPair( imod->lldChannel(ipair) );
	      pair<uint16_t,uint16_t> fed_channel = pair<uint16_t,uint16_t>( *ifed, fed_ch );
	      const_cast<SiStripModule&>(*imod).fedCh( addr.first, fed_channel );
	      ifed++;
	      
	    }
	    
	  }
	}
      }
    }
  }

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Finished assigning \"dummy\" FED ids/channels to constructed modules...";

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Cumulative time [s]: " 
    << time(NULL) - SiStripFedCablingBuilderFromDb::timer_;
 
  // ---------- Assign "dummy" devices to remaining FED ids/chans ----------

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Assigning \"dummy\" devices to remaining FED ids/channels...";

  uint16_t module = 0;
  bool complete = false;
  while ( !complete ) { 
    for ( uint16_t lld = sistrip::LLD_CHAN_MIN; 
	  lld <= sistrip::LLD_CHAN_MAX; lld++ ) {
      
      // Check FED id and channel
      if ( ifed == fed_ids.end() ) { fed_ch++; ifed = fed_ids.begin(); } 
      if ( fed_ch == 96 ) {
	edm::LogWarning(mlCabling_)
	  << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
	  << " All FED channels are now cabled!";
	complete = true;
	break;
      }

      // Create "invalid" connection with dummy FED id/channel
      if ( fecRing( module ) > 8 ) { 
	edm::LogWarning(mlTest_) 
	  << "TESTROB " 
	  << fecSlot( module ) << " " 
	  << fecRing( module ) << " " 
	  << ccuAddr( module ) << " " 
	  << ccuChan( module );
      }
      FedChannelConnection temp( sistrip::invalid_,
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
      fec_cabling.addDevices( temp );
      ifed++;
      
    } 
    module++;
  }

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Finished assigning \"dummy\" devices to remaining FED ids/channels...";

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Cumulative time [s]: " 
    << time(NULL) - SiStripFedCablingBuilderFromDb::timer_;

  // ---------- Assign DCU and DetIds and then FED cabling ----------
  
  assignDcuAndDetIds( fec_cabling, cached_map, used_map );
  
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
								SiStripFecCabling& fec_cabling,
								SiStripConfigDb::DcuDetIdMap& used_map ) {
  edm::LogVerbatim(mlCabling_)
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Building FEC cabling object from DetIds...";
  
  // ---------- Some initialization ----------
  used_map.clear();
  // fec_cabling.clear();
  
  // chans_per_ring = chans_per_ccu * ccus_per_ring = 100 (TOB gives lower limit of 60)
  uint32_t chans_per_ccu  = 10; 
  uint32_t ccus_per_ring  = 10;
  uint32_t rings_per_fec  = 8;
  uint32_t fecs_per_crate = 11;

  // ---------- Retrieve necessary descriptions from database ----------
  
  SiStripConfigDb::DcuDetIdMap cached_map = db->getDcuDetIdMap();
  if ( cached_map.empty() ) {
    edm::LogError(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " Unable to build FEC cabling!"
      << " DcuDetIdMap vector is empty!";
    return;
  }

  // ---------- Populate FEC cabling object with DCU, DetId and "dummy" control info ----------

  uint32_t imodule = 0;
  SiStripConfigDb::DcuDetIdMap::iterator iter;
  for ( iter = cached_map.begin(); iter != cached_map.end(); iter++ ) {
    uint16_t fec_crate = ( imodule / ( chans_per_ccu * ccus_per_ring * rings_per_fec * fecs_per_crate ) ) + 1;
    uint16_t fec_slot  = ( imodule / ( chans_per_ccu * ccus_per_ring * rings_per_fec ) ) % fecs_per_crate + 1;
    uint16_t fec_ring  = ( imodule / ( chans_per_ccu * ccus_per_ring ) ) % rings_per_fec + 1;
    uint16_t ccu_addr  = ( imodule / ( chans_per_ccu) ) % ccus_per_ring + 1;
    uint16_t ccu_chan  = ( imodule ) % chans_per_ccu + 26;
    
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
      FedChannelConnection conn( fec_crate, 
				 fec_slot, 
				 fec_ring, 
				 ccu_addr, 
				 ccu_chan, 
				 32+(2*ipair), 33+(2*ipair), // APV addresses
				 dcu_id, det_id, npairs,
				 0, 0, // FED id and channel
				 length,
				 true, true, true, true );
      fec_cabling.addDevices( conn );
    } 

    // --- Construct new TkDcuInfo object ---

    TkDcuInfo* dcu_info = new TkDcuInfo( dcu_id,
					 det_id,
					 length,
					 2*npairs );
    used_map[dcu_id] = dcu_info;

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
	      SiStripModule::FedChannel fed_channel = SiStripModule::FedChannel( fed_id, fed_ch );
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
							 SiStripConfigDb::DcuDetIdMap& in,
							 SiStripConfigDb::DcuDetIdMap& out ) {
  
  // ---------- Check if entries found in DCU-DetId map ----------

  if ( in.empty() ) { 
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No entries in DCU-DetId map!";
  }

  // ---------- Assign DCU and DetId to Modules in FEC cabling object ----------

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Assigning DCU ids and DetIds to constructed modules...";
  
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

	      SiStripConfigDb::DcuDetIdMap::iterator iter = in.find( module.dcuId() );
	      if ( iter != in.end() ) { 

		// --- Assign DetId and set nApvPairs based on APVs found in given Module ---
		
		module.detId( iter->second->getDetId() ); 
		module.nApvPairs(0); 
		
		// --- Check number of APV pairs is valid and consistent with cached map ---

		if ( module.nApvPairs() != 2 &&
		     module.nApvPairs() != 3 ) { 
		  stringstream ss1;
		  ss1 << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		      << " Module with DCU id 0x" 
		      << hex << setw(8) << setfill('0') << module.dcuId() << dec
		      << " and DetId 0x"
		      << hex << setw(8) << setfill('0') << module.detId() << dec
		      << " has unexpected number of APV pairs (" 
		      << module.nApvPairs()
		      << "). Some APV pairs may have not been detected by the FEC scan."
		      << " Setting to value found in static map (" 
		      << iter->second->getApvNumber()/2 << ")...";
		  edm::LogWarning(mlCabling_) << ss1.str();
		  module.nApvPairs( iter->second->getApvNumber()/2 ); 
		}
		
		if ( module.nApvPairs() < iter->second->getApvNumber()/2 ) {
		  stringstream ss2;
		  ss2 << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		      << " Module with DCU id 0x" 
		      << hex << setw(8) << setfill('0') << module.dcuId() << dec
		      << " and DetId 0x"
		      << hex << setw(8) << setfill('0') << module.detId() << dec
		      << " has number of APV pairs (" 
		      << module.nApvPairs()
		      << ") that does not match value found in DCU-DetId map (" 
		      << iter->second->getApvNumber()/2 
		      << "). Some APV pairs may have not been detected by"
		      << " the FEC scan or the DCU-DetId map may be incorrect."
		      << " Setting to value found in static map ("
		      << iter->second->getApvNumber()/2 << ")...";
		  edm::LogWarning(mlCabling_) << ss2.str();
		  module.nApvPairs( iter->second->getApvNumber()/2 ); 
		}
		
		// --- Add TkDcuInfo object to "used" map and remove from cached map ---

		out[module.dcuId()] = iter->second;
		in.erase( iter );

	      } // Set for DCU in static table
	    } // Check for null DetId

	  } // Module loop
	} // CCU loop
      } // FEC ring loop
    } // FEC loop
  } // FEC crate loop
  
  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Finished assigning DCU ids and DetIds to constructed modules...";

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Cumulative time [s]: " 
    << time(NULL) - SiStripFedCablingBuilderFromDb::timer_;

  // ---------- "Randomly" assign DetIds to Modules with DCU ids not found in static table ----------

  edm::LogVerbatim(mlCabling_) 
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
	      SiStripConfigDb::DcuDetIdMap::iterator iter = in.find( module.dcuId() );
	      if ( iter == in.end() ) { 
				
		// --- Search for "random" module with consistent number of APV pairs ---
		
		SiStripConfigDb::DcuDetIdMap::iterator idcu;
		if ( in.empty() ) { idcu = in.end(); }
		else {
		  idcu = in.begin();
		  while ( idcu != in.end() ) { 
		    if ( static_cast<uint32_t>(idcu->second->getApvNumber()) == 
			 static_cast<uint32_t>(2*module.nApvPairs()) ) { continue; }
		    idcu++; 
		  }
		} 
		
		// --- Assign "random" DetId if number of APV pairs is consistent ---
		
		if ( idcu != in.end() ) {
		  
		  module.detId( idcu->second->getDetId() );
		  TkDcuInfo* dcu_info = new TkDcuInfo( module.dcuId(),
						       idcu->second->getDetId(),
						       idcu->second->getFibreLength(),
						       idcu->second->getApvNumber() );
		  out[module.dcuId()] = dcu_info;
		  in.erase( idcu );
		  
		  stringstream ss;
		  ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		     << " Did not find module with DCU id 0x"
		     << hex << setw(8) << setfill('0') << module.dcuId() << dec
		     << " in DCU-DetId map!" << endl
		     << " Assigned 'random' DetId 0x"
		     << hex << setw(8) << setfill('0') << module.detId() << dec;
		  edm::LogWarning(mlCabling_) << ss.str();
		  
		} else { // --- Else, assign "dummy" DetId based on counter ---
		  
		  // If no match found, then assign DetId using incremented counter
		  module.detId( detid );
		  TkDcuInfo* dcu_info = new TkDcuInfo( module.dcuId(),
						       detid, 
						       0.,
						       2*module.nApvPairs() );
		  out[module.dcuId()] = dcu_info;
		  detid++;

		  stringstream ss;
		  if ( in.empty() ) {
		    ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		       << " Did not find module with DCU id 0x"
		       << hex << setw(8) << setfill('0') << module.dcuId() << dec
		       << " in DCU-DetId map!" 
		       << " Could not assign 'random' DetId as DCU-DetID map is empty!"
		       << " Assigned DetId based on incremented counter, with value 0x"
		       << hex << setw(8) << setfill('0') << module.detId() << dec;
		  } else {
		    ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		       << " Did not find module with DCU id 0x"
		       << hex << setw(8) << setfill('0') << module.dcuId() << dec
		       << " in DCU-DetId map!" 
		       << " Could not assign 'random' DetId as no modules had appropriate number of APV pairs ("
		       << module.nApvPairs()
		       << "). Assigned DetId based on incremented counter, with value 0x"
		       << hex << setw(8) << setfill('0') << module.detId() << dec;
		  }
		  edm::LogWarning(mlCabling_) << ss.str();

		}
	      }

	    } // Check for null DetId
	  } // Module loop
	} // CCU loop
      } // FEC ring loop
    } // FEC loop
  } // FEC crate loop

  edm::LogVerbatim(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Finished assigning \"random\" DetIds to modules with DCU ids not found in static table...";
  
  edm::LogVerbatim(mlCabling_) 
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
