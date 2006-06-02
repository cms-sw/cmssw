// Last commit: $Id$
// Latest tag:  $Name$
// Location:    $Source$

#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h" //@@ necessary?
#include <vector>
#include <string>
#include <cstdlib>

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilder::SiStripFedCablingBuilder() {
  edm::LogInfo ("FedCabling") << "[SiStripFedCablingBuilder::SiStripFedCablingBuilder]"
			      << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilder::~SiStripFedCablingBuilder() {
  edm::LogInfo("FedCabling") << "[SiStripFedCablingBuilder::~SiStripFedCablingBuilder]"
			     << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void SiStripFedCablingBuilder::createFedCabling( SiStripConfigDb* const db,
						 SiStripFedCabling& fed_cabling,
						 SiStripConfigDb::DcuIdDetIdMap& output_map ) {
}

// -----------------------------------------------------------------------------
/** 
    Builds the SiStripFedCabling conditions object that is available
    via the EventSetup interface. The object contains the full
    FedChannel-DcuId-DetId mapping information.

    The map is built using information cached by the SiStripConfigDb
    object, comprising: 1) the FED channel connections, as found in
    the "module.xml" file; 2) and DcuId-DetId mapping, as found in the
    "dcuinfo.xml" file. If any information is missing, the method
    provides "dummy" values.
    
    Methodology:

    The FEC cabling object is built using FED channel connection
    objects (ie, from "module.xml"). 

    If the DcuId (provided by the hardware device descriptions) is
    null, a dummy value is provided, based on the control key.
    
    The DcuId-DetId map (ie, from "dcuinfo.xml") is queried for a
    matching DcuId. If found, the DetId and ApvPairs are updated. If
    not, a "random" DetId within the DcuId-DetId map is assigned. Note
    that a check is made on the number of APV pairs before the DetId
    is assigned. If no appropriate match is found, the DetId is
    assigned a value using an incremented counter (starting from
    0xFFFF).
    
    All DcuId-DetId mappings are accumulated in a new map, and this
    modified map is returned by the method.
*/
void SiStripFedCablingBuilder::createFedCablingFromFedConnections( SiStripConfigDb* const db,
								   SiStripFedCabling& fed_cabling,
								   SiStripConfigDb::DcuIdDetIdMap& new_map ) {
  LogDebug("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]";
  
  // Clear new DcuId-DetId map
  new_map.clear();

  // Retrieve FED connections and check if any exist
  SiStripConfigDb::FedConnections conns = db->getFedConnections();
  if ( conns.empty() ) { 
    edm::LogError("ConfigDb") << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]"
			      << " No FedConnection objects found!";
    return;
  }

  // Create FEC cabling object to be populated
  SiStripFecCabling fec_cabling;
  
  // Retrieve cached DcuId-DetId map
  SiStripConfigDb::DcuIdDetIdMap cached_map = db->getDcuIdDetIdMap();

  // Iterate through FedConnections and retrieve all connection info
  SiStripConfigDb::FedConnections::const_iterator ifed;
  for ( ifed = conns.begin(); ifed != conns.end(); ifed++ ) {
    
    // Retrieve hardware addresses
    uint16_t fec_crate = 0; //@@ needs implementing!
    uint16_t fec_slot  = static_cast<uint16_t>( (*ifed)->getSlot() );
    uint16_t fec_ring  = static_cast<uint16_t>( (*ifed)->getRing() );
    uint16_t ccu_addr  = static_cast<uint16_t>( (*ifed)->getCcu() );
    uint16_t ccu_chan  = static_cast<uint16_t>( (*ifed)->getI2c() );
    uint16_t apv0      = static_cast<uint16_t>( (*ifed)->getApv() );
    uint16_t apv1      = apv0 + 1; //@@ needs implementing!
    uint32_t dcu_id    = static_cast<uint32_t>( (*ifed)->getDcuHardId() );
    
    // Retrieve FED id and channel
    uint16_t fed_id    = static_cast<uint16_t>( (*ifed)->getFedId() );
    uint16_t fed_ch    = static_cast<uint16_t>( (*ifed)->getFedChannel() );
    
    // Create "FED channel connection" object
    FedChannelConnection conn( fec_crate, fec_slot, fec_ring, ccu_addr, ccu_chan,
			       apv0, apv1,
			       dcu_id, 0, 0, // null values for DetId and ApvPairs 
			       fed_id, fed_ch );
    // Add object to FEC cabling
    fec_cabling.addDevices( conn );
    
  }
  
  // Iterate through FEC cabling
  uint32_t det_id = 0xFFFF;
  for ( vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin(); icrate != fec_cabling.crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    SiStripModule& module = const_cast<SiStripModule&>(*imod);
	    
	    // Check for null DcuId. If zero, set equal to control key
	    if ( !module.dcuId() ) { 
	      edm::LogWarning("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]"
					    << " Null DcuId found in FEC cabling!"
					    << " Providing 'dummy' DcuId based on control key...";
	      uint32_t module_key = SiStripControlKey::key( icrate->fecCrate(),
							    ifec->fecSlot(), 
							    iring->fecRing(), 
							    iccu->ccuAddr(), 
							    imod->ccuChan() );
	      module.dcuId( module_key );
	    }
	    
	    // Check for null DetId
	    if ( !module.detId() ) { 
	      edm::LogWarning("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]"
					    << " Null DetId found in FEC cabling!"
					    << " Providing 'dummy' DcuId-DetId mapping...";
	      // Set number of APV pairs based on number found in module
	      module.nApvPairs(0); 
	      // Search for DcuId in map and correct number of APV pairs
	      SiStripConfigDb::DcuIdDetIdMap::iterator iter = cached_map.find( module.dcuId() );
	      if ( iter != cached_map.end() &&
		   module.nApvPairs() == iter->second->getApvNumber()/2 ) { 
		// If found, add TkDcuInfo object to new map and remove from cached map
		new_map[module.dcuId()] = iter->second;
		//cached_map.erase( iter );
	      } else {
		// If no match found, iterate through cached map and find DetId with correct number of APV pairs
		SiStripConfigDb::DcuIdDetIdMap::iterator idcu = cached_map.begin();
		while ( static_cast<uint32_t>(idcu->second->getApvNumber()) != static_cast<uint32_t>(2*module.nApvPairs()) &&
			idcu != cached_map.end() ) { idcu++; }
		if ( idcu != cached_map.end() ) {
		  // If match found, then create TkDcuInfo object in new map
		  TkDcuInfo* dcu_info = new TkDcuInfo( module.dcuId(),
						       idcu->second->getDetId(),
						       idcu->second->getFibreLength(),
						       idcu->second->getApvNumber() );
		  new_map[module.dcuId()] = dcu_info;
		  //cached_map.erase( idcu );
		} else {
		  // If no match found, then assign DetId using incremented counter
		  edm::LogWarning("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]"
						<< " Cannot assign DetId to DcuId!";
		  TkDcuInfo* dcu_info = new TkDcuInfo( module.dcuId(),
						       det_id, 
						       0.,
						       2*module.nApvPairs() );
		  new_map[module.dcuId()] = dcu_info;
		  det_id++;
		}
	      }
	    }

	    LogDebug("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]"
				   << " Setting DcuId/DetId/nApvPairs = " 
				   << module.dcuId() << "/" 
				   << module.detId() << "/" 
				   << module.nApvPairs()
				   << " for Crate/FEC/Ring/CCU/Module " 
				   << icrate->fecCrate() << "/" 
				   << ifec->fecSlot() << "/" 
				   << iring->fecRing() << "/" 
				   << iccu->ccuAddr() << "/" 
				   << imod->ccuChan();
	    
	  }
	}
      }
    }
  }

  // Warn if not all DetIds have been assigned to DcuIds
  if ( !cached_map.empty() ) {
    stringstream ss;
    ss << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]"
       << " " << cached_map.size() << " DetIds have not been assigned to a DcuId!";
    edm::LogWarning("FedCabling") << ss.str();
  }
  
  // Generate FED cabling from FEC cabling object
  SiStripFedCablingBuilder::getFedCabling( fec_cabling, fed_cabling );
  
}

// -----------------------------------------------------------------------------
/** 
    Builds the SiStripFedCabling conditions object that is available
    via the EventSetup interface. The object contains the full
    FedChannel-DcuId-DetId mapping information.
    
    This method is typically used when the FED connections (ie,
    "module.xml" file) does not exist, such as prior to the FED
    cabling or "bare connection" procedure.

    The map is built using information cached by the SiStripConfigDb
    object, comprising: 1) the hardware device descriptions, as found
    in the "fec.xml" file; 2) and DcuId-DetId mapping, as found in the
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
    
    The DcuId-DetId map (ie, from "dcuinfo.xml") is queried for a
    matching DcuId. If found, the DetId and ApvPairs are updated. If
    not, a "random" DetId within the DcuId-DetId map is assigned. Note
    that a check is made on the number of APV pairs before the DetId
    is assigned. If no appropriate match is found, the DetId is
    assigned a value using an incremented counter (starting from
    0xFFFF).

    All DcuId-DetId mappings are accumulated in a new map, and this
    modified map is returned by the method.
*/
void SiStripFedCablingBuilder::createFedCablingFromDevices( SiStripConfigDb* const db,
							    SiStripFedCabling& fed_cabling,
							    SiStripConfigDb::DcuIdDetIdMap& new_map ) {
  LogDebug("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromDevices]";
  
  // Clear new DcuId-DetId map
  new_map.clear();

  // Create FEC cabling object to be populated
  SiStripFecCabling fec_cabling;
  
  // Retrieve APV descriptions and populate FEC cabling map 
  const SiStripConfigDb::DeviceDescriptions& apv_desc = db->getDeviceDescriptions( APV25 );
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
  edm::LogInfo("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromDevices]"
			     << " Created FEC cabling based on " << apv_desc.size()
			     << " APV descriptions extracted from config DB...";
  
  // Retrieve DCU descriptions and populate FEC cabling map 
  const SiStripConfigDb::DeviceDescriptions& dcu_desc = db->getDeviceDescriptions( DCU );
  SiStripConfigDb::DeviceDescriptions::const_iterator idcu;
  for ( idcu = dcu_desc.begin(); idcu != dcu_desc.end(); idcu++ ) {
    SiStripConfigDb::DeviceAddress addr = db->deviceAddress(**idcu);
    dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idcu );
    if ( !dcu ) {
      edm::LogError("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromDevices]"
				  << " Null pointer to dcuDescription!";
      continue;
    }
    FedChannelConnection conn( addr.fecCrate_, 
			       addr.fecSlot_, 
			       addr.fecRing_, 
			       addr.ccuAddr_, 
			       addr.ccuChan_,
			       0, 0, //@@ APV I2C addresses (not used)
			       dcu->getDcuHardId() ); 
    fec_cabling.dcuId( conn );
  }
  edm::LogInfo("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromDevices]"
			     << " Extracted " << dcu_desc.size()
			     << " DCU hardware ids from config DB";
  
  // Estimate number of FEDs required to cable entire control system
  const NumberOfDevices& devices = fec_cabling.countDevices();
  uint16_t required_feds = devices.nApvPairs_/96 + 1;
  
  // Check if there are sufficient FED ids to cable entire control system
  vector<uint16_t> fed_ids = db->getFedIds();
  if ( fed_ids.size() < required_feds ) {
    edm::LogInfo("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromDevices]"
			       << " Insufficient number of FED descriptions (" << fed_ids.size()
			       << ") in config DB to cable all APV pairs (" << devices.nApvPairs_
			       << "). Generating random FED ids...";
    // If insufficient FEDs, add "dummy" fed_ids
    uint16_t fed_id = 50;
    while ( fed_ids.size() < required_feds ) {
      vector<uint16_t>::iterator ifed = find( fed_ids.begin(), fed_ids.end(), fed_id );
      if ( ifed == fed_ids.end() ) { fed_ids.push_back( fed_id ); }
      fed_id++;
    }
  }
  
  // Retrieve cached DcuId-DetId map
  SiStripConfigDb::DcuIdDetIdMap cached_map = db->getDcuIdDetIdMap();

  // Iterate through control system
  vector<uint16_t>::iterator ifed = fed_ids.begin();
  uint32_t fed_ch = 0;
  uint32_t det_id = 0xFFFF;
  for ( vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin(); icrate != fec_cabling.crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    SiStripModule& module = const_cast<SiStripModule&>(*imod);
	    
	    // Provide dummy cabling to FEDs
	    if ( 96-fed_ch < imod->nApvPairs() ) { ifed++; fed_ch = 0; } // move to next FED
	    if ( ifed == fed_ids.end() ) {
	      edm::LogError("FedCabling") << "[SiStripFedCablingBuilderFromDb::createFecCablingFromConnections]"
					  << " Insufficient FEDs to cabling entire control system!";
	      break;
	    }
	    for ( uint16_t ipair = 0; ipair < imod->nApvPairs(); ipair++ ) {
	      pair<uint16_t,uint16_t> addr = imod->activeApvPair( (*imod).lldChannel(ipair) );
	      pair<uint16_t,uint16_t> fed_channel = pair<uint16_t,uint16_t>( *ifed, fed_ch );
	      const_cast<SiStripModule&>(*imod).fedCh( addr.first, fed_channel );
	      fed_ch++;
	    }

	    // Check for null DcuId. If zero, set equal to control key
	    if ( !module.dcuId() ) { 
	      edm::LogWarning("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]"
					    << " Null DcuId found in FEC cabling!"
					    << " Providing 'dummy' DcuId based on control key...";
	      uint32_t module_key = SiStripControlKey::key( 0, // fec crate 
							    ifec->fecSlot(), 
							    iring->fecRing(), 
							    iccu->ccuAddr(), 
							    imod->ccuChan() );
	      module.dcuId( module_key );
	    }
	  
	    // Check for null DetId
	    if ( !module.detId() ) { 
	      edm::LogWarning("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]"
					    << " Null DetId found in FEC cabling!"
					    << " Providing 'dummy' DcuId-DetId mapping...";
	      // Set number of APV pairs based on number found in module
	      module.nApvPairs(0); 
	      // Search for DcuId in map and correct number of APV pairs
	      SiStripConfigDb::DcuIdDetIdMap::iterator iter = cached_map.find( module.dcuId() );
	      if ( iter != cached_map.end() &&
		   module.nApvPairs() == iter->second->getApvNumber()/2 ) { 
		// If found, add TkDcuInfo object to new map and remove from cached map
		new_map[module.dcuId()] = iter->second;
		//cached_map.erase( iter );
	      } else {
		// If no match found, iterate through cached map and find DetId with correct number of APV pairs
		SiStripConfigDb::DcuIdDetIdMap::iterator idcu = cached_map.begin();
		while ( static_cast<uint32_t>(idcu->second->getApvNumber()) != static_cast<uint32_t>(2*module.nApvPairs()) &&
			idcu != cached_map.end() ) { idcu++; }
		if ( idcu != cached_map.end() ) {
		  // If match found, then create TkDcuInfo object in new map
		  TkDcuInfo* dcu_info = new TkDcuInfo( module.dcuId(),
						       idcu->second->getDetId(),
						       idcu->second->getFibreLength(),
						       idcu->second->getApvNumber() );
		  new_map[module.dcuId()] = dcu_info;
		  //cached_map.erase( idcu );
		} else {
		  // If no match found, then assign DetId using incremented counter
		  edm::LogWarning("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]"
						<< " Cannot assign DetId to DcuId!";
		  TkDcuInfo* dcu_info = new TkDcuInfo( module.dcuId(),
						       det_id, 
						       0.,
						       2*module.nApvPairs() );
		  new_map[module.dcuId()] = dcu_info;
		  det_id++;
		}
	      }
	    }

	    LogDebug("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]"
				   << " Setting DcuId/DetId/nApvPairs = " 
				   << module.dcuId() << "/" 
				   << module.detId() << "/" 
				   << module.nApvPairs()
				   << " for Crate/FEC/Ring/CCU/Module " 
				   << icrate->fecCrate() << "/" 
				   << ifec->fecSlot() << "/" 
				   << iring->fecRing() << "/" 
				   << iccu->ccuAddr() << "/" 
				   << imod->ccuChan();
	  
	  }
	}
      }
    }
  }

  // Warn if not all DetIds have been assigned to DcuIds
  if ( !cached_map.empty() ) {
    stringstream ss;
    ss << "[SiStripFedCablingBuilder::createFedCablingFromFedConnections]"
       << " " << cached_map.size() << " DetIds have not been assigned to a DcuId!";
    edm::LogWarning("FedCabling") << ss.str();
  }

  // Generate FED cabling from FEC cabling object
  SiStripFedCablingBuilder::getFedCabling( fec_cabling, fed_cabling );

}

// -----------------------------------------------------------------------------
/**
   Builds the SiStripFedCabling conditions object that is available
   via the EventSetup interface. The object contains the full
   FedChannel-DcuId-DetId mapping information.
    
   This method is typically used when only the DcuId-DetId map (ie,
   from "dcuinfo.xml") exists and the FED connections (ie,
   "module.xml" file) and device descriptions (ie, from "fec.xml")
   are both missing.
    
   The map is built using the DcuId-DetId map that is cached by the
   SiStripConfigDb object. As a minimum, the map should contain
   values within both the DetId and APpvPair fields, but if any
   information is missing, the method provides "dummy" values.
    
   Methodology:

   The FEC cabling object is built using the DcuId-DetId map (ie,
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

   All DcuId-DetId mappings are accumulated in a new map, and this
   modified map is returned by the method.
*/
void SiStripFedCablingBuilder::createFedCablingFromDetIds( SiStripConfigDb* const db,
							   SiStripFedCabling& fed_cabling,
							   SiStripConfigDb::DcuIdDetIdMap& new_map ) {
  LogDebug("FedCabling") << "[SiStripFedCablingBuilder::createFedCablingFromDetIds]";

  // Clear new DcuId-DetId map
  new_map.clear();

  // Create FEC cabling object to be populated
  SiStripFecCabling fec_cabling;
  
  // Retrieve through cached DcuId-DetId map
  SiStripConfigDb::DcuIdDetIdMap cached_map = db->getDcuIdDetIdMap();
  
  // Some constants (TOB gives lower limit on chans_per_ring = 60)
  uint32_t chans_per_ccu  = 10; 
  uint32_t ccus_per_ring  = 10;
  uint32_t rings_per_fec  = 8;
  uint32_t fecs_per_crate = 11;
  
  // Iterate through cached DcuId-DetId map and assign DcuId (based on control key)
  uint32_t imodule = 0;
  SiStripConfigDb::DcuIdDetIdMap::iterator iter;
  for ( iter = cached_map.begin(); iter != cached_map.end(); iter++ ) {
    uint16_t fec_crate = ( imodule / ( chans_per_ccu * ccus_per_ring * rings_per_fec * fecs_per_crate ) ) + 1;
    uint16_t fec_slot  = ( imodule / ( chans_per_ccu * ccus_per_ring * rings_per_fec ) ) % fecs_per_crate + 1;
    uint16_t fec_ring  = ( imodule / ( chans_per_ccu * ccus_per_ring ) ) % rings_per_fec + 1;
    uint16_t ccu_addr  = ( imodule / ( chans_per_ccu) ) % ccus_per_ring + 1;
    uint16_t ccu_chan  = ( imodule ) % chans_per_ccu + 26;

    uint32_t dcu_id = iter->second->getDcuHardId();
    uint32_t det_id = iter->second->getDetId();
    uint16_t npairs = iter->second->getApvNumber()/2;
    uint16_t length = (uint16_t) iter->second->getFibreLength(); //@@ should be double!

    // If DcuId is null, set equal to control key
    if ( !dcu_id ) {
      dcu_id = SiStripControlKey::key( fec_crate,
				       fec_slot,
				       fec_ring,
				       ccu_addr,
				       ccu_chan );
    }
    // If DetId is null, set equal to incremented counter (starting from 0xFFFF)
    if ( !det_id ) { det_id = 0xFFFF + imodule; } 
    // If number of APV pairs is null, set to random value (2 or 3) 
    if ( !npairs ) { npairs = rand()/2 ? 2 : 3; }
    
    for ( uint16_t ipair = 0; ipair < npairs; ipair++ ) {
      // Create FED channel connection object
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
    
    // Iterate module counter
    imodule++;
    // Create new TkDcuInfo object
    TkDcuInfo* dcu_info = new TkDcuInfo( dcu_id,
					 det_id,
					 length,
					 2*npairs );
    new_map[dcu_id] = dcu_info;
  }
  
  // Iterate through control system and provide dummy FED cabling
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

  // Generate FED cabling from FEC cabling object
  SiStripFedCablingBuilder::getFedCabling( fec_cabling, fed_cabling );
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripFedCablingBuilder::getFedCabling( const SiStripFecCabling& fec_cabling, 
					      SiStripFedCabling& fed_cabling ) {
  vector<FedChannelConnection> conns;
  fec_cabling.connections( conns );
  fed_cabling.buildFedCabling( conns );
}

// -----------------------------------------------------------------------------
/** */
void SiStripFedCablingBuilder::getFecCabling( const SiStripFedCabling& fed_cabling, 
					      SiStripFecCabling& fec_cabling ) {
  fec_cabling.buildFecCabling( fed_cabling );
}









