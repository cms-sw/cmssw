// Last commit: $Id: SiStripFedCablingBuilderFromDb.cc,v 1.23 2006/10/10 14:37:36 bainbrid Exp $
// Latest tag:  $Name: TIF_101006 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/src/SiStripFedCablingBuilderFromDb.cc,v $
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <cstdlib>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderFromDb::SiStripFedCablingBuilderFromDb( const edm::ParameterSet& pset ) 
  : SiStripFedCablingESSource( pset ),
    db_(0),
    partitions_( pset.getUntrackedParameter< vector<string> >( "Partitions", vector<string>() ) ) //@@@ use this????
{
  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
  
  if ( pset.getUntrackedParameter<bool>( "UsingDb", true ) ) {
    // Using database 
    db_ = new SiStripConfigDb( pset.getUntrackedParameter<string>("ConfDb",""),
			       pset.getUntrackedParameter<string>("Partition",""),
			       pset.getUntrackedParameter<unsigned int>("MajorVersion",0),
			       pset.getUntrackedParameter<unsigned int>("MinorVersion",0) );
  } else {
    // Using xml files
    db_ = new SiStripConfigDb( pset.getUntrackedParameter<string>("InputModuleXml",""),
			       pset.getUntrackedParameter<string>("InputDcuInfoXml",""),
			       pset.getUntrackedParameter< vector<string> >( "InputFecXml", vector<string>(1,"") ),
			       pset.getUntrackedParameter< vector<string> >( "InputFedXml", vector<string>(1,"") ), 
			       pset.getUntrackedParameter<string>("OutputModuleXml","/tmp/module.xml"),
			       pset.getUntrackedParameter<string>("OutputDcuInfoXml","/tmp/dcuinfo.xml"),
			       pset.getUntrackedParameter<string>( "OutputFecXml", "/tmp/fec.xml" ),
			       pset.getUntrackedParameter<string>( "OutputFedXml", "/tmp/fed.xml" ) );
  }
  
  // Establish connection
  db_->openDbConnection();
  
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderFromDb::~SiStripFedCablingBuilderFromDb() {
  LogTrace(mlCabling_)
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Destructing object...";
  if ( db_ ) { 
    db_->closeDbConnection();
    delete db_;
  } 
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCabling* SiStripFedCablingBuilderFromDb::makeFedCabling() {
  
  // Build FEC cabling object
  SiStripFecCabling fec_cabling;
  SiStripConfigDb::DcuDetIdMap dcu_detid_map;
  buildFecCabling( db_, fec_cabling, dcu_detid_map ); 
  
  // Build FED cabling object 
  SiStripFedCabling* fed_cabling = new SiStripFedCabling();
  getFedCabling( fec_cabling, *fed_cabling );
  
  // Call virtual method that writes FED cabling object to conditions DB
  writeFedCablingToCondDb( *fed_cabling );

  // Prints FED cabling
  stringstream ss;
  ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]" 
     << " Printing cabling map..." << endl 
     << *fed_cabling;
  LogTrace(mlCabling_) << ss.str();
  
  return fed_cabling;
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripFedCablingBuilderFromDb::buildFecCabling( SiStripConfigDb* const db,
						      SiStripFecCabling& fec_cabling,
						      SiStripConfigDb::DcuDetIdMap& new_map ) {

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
  const NumberOfDevices& devs = fec_cabling.countDevices();
  stringstream ss;
  ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
     << " Built SiStripFecCabling object with following devices:" 
     << endl << devs;
  LogTrace(mlCabling_) << ss.str() << endl;
  
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
  LogTrace(mlCabling_)
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
      << " No entries in FedConnections vector!";
    return;
  }
  
  SiStripConfigDb::DcuDetIdMap cached_map = db->getDcuDetIdMap();
  if ( cached_map.empty() ) { 
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No entries in DCU-DetId map!";
  }
  
  // ---------- Populate FEC cabling object with retrieved info ----------

  SiStripConfigDb::FedConnections::const_iterator ifed = conns.begin();
  for ( ; ifed != conns.end(); ifed++ ) {
    uint16_t fec_crate = static_cast<uint16_t>( (*ifed)->getFecInstance() ); //@@ needs implementing!
    uint16_t fec_slot  = static_cast<uint16_t>( (*ifed)->getSlot() );
    uint16_t fec_ring  = static_cast<uint16_t>( (*ifed)->getRing() );
    uint16_t ccu_addr  = static_cast<uint16_t>( (*ifed)->getCcu() );
    uint16_t ccu_chan  = static_cast<uint16_t>( (*ifed)->getI2c() );
    uint16_t apv0      = static_cast<uint16_t>( (*ifed)->getApv() );
    uint16_t apv1      = apv0 + 1; //@@ needs implementing!
    uint32_t dcu_id    = static_cast<uint32_t>( (*ifed)->getDcuHardId() );
    uint32_t det_id    = static_cast<uint32_t>( (*ifed)->getDetId() );
    uint16_t npairs    = static_cast<uint16_t>( (*ifed)->getApvPairs() );
    uint16_t fed_id    = static_cast<uint16_t>( (*ifed)->getFedId() );
    uint16_t fed_ch    = static_cast<uint16_t>( (*ifed)->getFedChannel() );
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
  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
    << " Building FEC cabling object from device descriptions...";
  
  // ---------- Some initialization ----------

  used_map.clear();
  // fec_cabling.clear();
  
  // ---------- Retrieve device descriptions from database ----------

  SiStripConfigDb::DeviceDescriptions apv_desc;
  db->getDeviceDescriptions( apv_desc, APV25 );
  if ( apv_desc.empty() ) { 
    edm::LogWarning(mlCabling_) 
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No APV descriptions found!";
    return;
  }
  
  SiStripConfigDb::DeviceDescriptions dcu_desc;
  db->getDeviceDescriptions( dcu_desc, DCU );
  if ( dcu_desc.empty() ) { 
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No DCU descriptions found!";
  }

  SiStripConfigDb::DcuDetIdMap cached_map = db->getDcuDetIdMap();
  if ( cached_map.empty() ) { 
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No entries in DCU-DetId map!";
  }
  
  vector<uint16_t> fed_ids = db->getFedIds();
  if ( fed_ids.empty() ) { 
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
      << " No FED descriptions found!";
  }

  // ---------- Populate FEC cabling object with retrieved info ----------

  SiStripConfigDb::DeviceDescriptions::const_iterator iapv;
  for ( iapv = apv_desc.begin(); iapv != apv_desc.end(); iapv++ ) {
    const SiStripConfigDb::DeviceAddress& addr = db->deviceAddress(**iapv);
    FedChannelConnection conn( addr.fecCrate_, 
			       addr.fecSlot_, 
			       addr.fecRing_, 
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
    FedChannelConnection conn( addr.fecCrate_, 
			       addr.fecSlot_, 
			       addr.fecRing_, 
			       addr.ccuAddr_, 
			       addr.ccuChan_,
			       0, 0, //@@ APV I2C addresses (not used)
			       dcu->getDcuHardId() ); 
    fec_cabling.dcuId( conn );
  }
  
  // ---------- Assign "dummy" FED ids/chans to Modules of FEC cabling object ----------

  vector<uint16_t>::iterator ifed = fed_ids.begin();
  uint32_t fed_ch = 0;
  for ( vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin(); icrate != fec_cabling.crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    
	    // Set number of APV pairs based on devices found 
	    const_cast<SiStripModule&>(*imod).nApvPairs(0); 
	    
	    // Provide dummy FED id/channel
	    for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {
	      if ( ifed == fed_ids.end() ) { fed_ch++; ifed = fed_ids.begin(); } // move to next FED channel
	      if ( fed_ch == 96 ) {
		edm::LogWarning(mlCabling_)
		  << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		  << " Insufficient FED channels to cable entire system!";
		break;
	      }
	      pair<uint16_t,uint16_t> addr = imod->activeApvPair( (*imod).lldChannel(ipair) );
	      pair<uint16_t,uint16_t> fed_channel = pair<uint16_t,uint16_t>( *ifed, fed_ch );
	      const_cast<SiStripModule&>(*imod).fedCh( addr.first, fed_channel );
	      ifed++;
	    }
	    
	  }
	}
      }
    }
  }
  
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
  LogTrace(mlCabling_)
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
      dcu_id = SiStripFecKey::key( fec_crate,
				   fec_slot,
				   fec_ring,
				   ccu_addr,
				   ccu_chan );
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

  // ---------- Assign DCU and DetId to Modules in FEC cabling object ----------
  
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
	      SiStripFecKey::Path path( icrate->fecCrate(),
					ifec->fecSlot(), 
					iring->fecRing(), 
					iccu->ccuAddr(), 
					imod->ccuChan() );
	      uint32_t module_key = SiStripFecKey::key( path );
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
		
		stringstream ss;
		ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		   << " Retrieved DetId 0x"
		   << hex << setw(8) << setfill('0') << module.detId() << dec
		   << " from static table for module with DCU id 0x"
		   << hex << setw(8) << setfill('0') << module.dcuId() << dec;
		LogTrace(mlCabling_) << ss.str();

		// --- Check number of APV pairs is valid and consistent with cached map ---

		if ( module.nApvPairs() != 2 &&
		     module.nApvPairs() != 3 ) { 
		  stringstream ss1;
		  ss1 << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		      << " Module with DCU id 0x" 
		      << hex << setw(8) << setfill('0') << module.dcuId() << dec
		      << " and DetId 0x"
		      << hex << setw(8) << setfill('0') << module.detId() << dec
		      << " has unexpected number of APV pairs (" << module.nApvPairs()
		      << ")." << endl
		      << " Setting to value found in static map (" << iter->second->getApvNumber()/2 << ")";
		  edm::LogWarning(mlCabling_) << ss1.str();
		  module.nApvPairs( iter->second->getApvNumber()/2 ); 
		}
		
		if ( module.nApvPairs() != iter->second->getApvNumber()/2 ) {
		  stringstream ss2;
		  ss2 << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		      << " Module with DCU id 0x" 
		      << hex << setw(8) << setfill('0') << module.dcuId() << dec
		      << " and DetId 0x"
		      << hex << setw(8) << setfill('0') << module.detId() << dec
		      << " has number of APV pairs (" << module.nApvPairs()
		      << " that does not match value found in static map (" << iter->second->getApvNumber()/2 
		      << ")." << endl
		      << " Setting to value found in static map.";
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
  
  // ---------- "Randomly" assign DetIds to Modules with DCU ids not found in static table ----------
  
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
		  ss << "[SiStripFedCablingBuilderFromDb::" << __func__ << "]"
		     << " Did not find module with DCU id 0x"
		     << hex << setw(8) << setfill('0') << module.dcuId() << dec
		     << " in DCU-DetId map!" 
		     << " Could not assign 'random' DetId as no modules had appropriate number of APV pairs ("
		     << module.nApvPairs()
		     << "). Assigned DetId based on incremented counter, with value 0x"
		     << hex << setw(8) << setfill('0') << module.detId() << dec;
		  edm::LogWarning(mlCabling_) << ss.str();

		}
	      }

	    } // Check for null DetId
	  } // Module loop
	} // CCU loop
      } // FEC ring loop
    } // FEC loop
  } // FEC crate loop
  
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








  
