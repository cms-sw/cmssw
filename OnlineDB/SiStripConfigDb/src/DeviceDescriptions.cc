// Last commit: $Id: DeviceDescriptions.cc,v 1.26 2008/04/29 11:57:05 bainbrid Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DeviceDescriptionsRange SiStripConfigDb::getDeviceDescriptions( std::string partition ) {

  // Check
  if ( ( !dbParams_.usingDbCache() && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache() && !databaseCache(__func__) ) ) { 
    return devices_.emptyRange(); 
  }
  
  try { 

    if ( !dbParams_.usingDbCache() ) { 

      SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().first;
      SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().second;
      for ( ; iter != jter; ++iter ) {
	
	if ( partition == "" || partition == iter->second.partitionName() ) {
	  
	  DeviceDescriptionsRange range = devices_.find( iter->second.partitionName() );
	  if ( range == devices_.emptyRange() ) {
	    
	    // Retrieve conections
	    std::vector<DeviceDescription*> tmp1;
	    deviceFactory(__func__)->getFecDeviceDescriptions( iter->second.partitionName(), 
							       tmp1,
							       iter->second.fecVersion().first,
							       iter->second.fecVersion().second,
							       false ); //@@ do not get DISABLED devices
	    
	    // Make local copy 
	    std::vector<DeviceDescription*> tmp2;
#ifdef USING_NEW_DATABASE_MODEL
	    FecFactory::vectorCopyI( tmp2, tmp1, true );
#else
	    tmp2 = tmp1;
#endif

	    // Add to cache
	    devices_.loadNext( iter->second.partitionName(), tmp2 );

	    // Some debug
	    DeviceDescriptionsRange range = devices_.find( iter->second.partitionName() );
	    std::stringstream ss;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Dowloaded " << range.size() 
	       << " device descriptions to local cache for partition \""
	       << iter->second.partitionName() << "\"" << std::endl;
	    ss << "[SiStripConfigDb::" << __func__ << "]"
	       << " Cache holds device descriptions for " 
	       << devices_.size() << " partitions.";
	    LogTrace(mlConfigDb_) << ss.str();

	  }
	  
	}
	
      }
      
    } else { // Using database cache
      
#ifdef USING_NEW_DATABASE_MODEL

      std::vector<DeviceDescription*>* tmp1 = databaseCache(__func__)->getDevices();

      if ( tmp1 ) { 
	
	// Make local copy 
	std::vector<DeviceDescription*> tmp2;
	FecFactory::vectorCopyI( tmp2, *tmp1, true );
	
	// Add to cache
	devices_.loadNext( "", tmp2 );

      } else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to DeviceDescriptions vector!";
      }

#endif // USING_NEW_DATABASE_MODEL
      
    }
    
  } catch (...) { handleException( __func__ ); }
  
  // Create range object
  uint16_t np = 0;
  uint16_t nc = 0;
  DeviceDescriptionsRange devs;
  if ( partition != "" ) { 
    devs = devices_.find( partition );
    np = 1;
    nc = devs.size();
  } else { 
    if ( !devices_.empty() ) {
      devs = DeviceDescriptionsRange( devices_.find( dbParams_.partitions().first->second.partitionName() ).begin(),
					devices_.find( (--(dbParams_.partitions().second))->second.partitionName() ).end() );
    } else { devs = devices_.emptyRange(); }
    np = devices_.size();
    nc = devs.size();
  }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Found " << nc << " device descriptions";
  if ( !dbParams_.usingDb() ) { ss << " in " << dbParams_.inputFecXmlFiles().size() << " 'fec.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache() )  { ss << " in " << np << " database partition(s)"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory() << "'"; } }
  if ( devices_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }
  
  return devs;

}

// -----------------------------------------------------------------------------
// 
//SiStripConfigDb::DeviceDescriptionsRange SiStripConfigDb::getDeviceDescriptions( DeviceType device_type, 
SiStripConfigDb::DeviceDescriptionsRange SiStripConfigDb::getDeviceDescriptions( DeviceType device_type, 
										   std::string partition ) {
  
  typedDevices_.clear();
  
  if ( ( !dbParams_.usingDbCache() && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache() && !databaseCache(__func__) ) ) { 
    return DeviceDescriptionsRange( typedDevices_.end(), typedDevices_.end() );
    //return std::make_pair( typedDevices_.end(), typedDevices_.end() );
  }
  
  try { 
    
    DeviceDescriptionsRange devs = SiStripConfigDb::getDeviceDescriptions( partition );
    
    if ( !devs.empty() ) {
      std::vector<DeviceDescription*> tmp( devs.begin(), devs.end() );
      typedDevices_ = FecFactory::getDeviceFromDeviceVector( tmp, device_type );
    }
    
  } catch (...) { handleException( __func__ ); }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Extracted " << typedDevices_.size() 
     << " device descriptions (for devices of type " 
     << deviceType( device_type ) << ")";
  
  return DeviceDescriptionsRange( typedDevices_.begin(), typedDevices_.end() );
  //return std::make_pair( typedDevices_.begin(), typedDevices_.end() );
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::addDeviceDescriptions( std::string partition, std::vector<DeviceDescription*>& devs ) {

  if ( !deviceFactory(__func__) ) { return; }

  if ( partition.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition string is empty,"
       << " therefore cannot add device descriptions to local cache!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  if ( devs.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Vector of device descriptions is empty,"
       << " therefore cannot add device descriptions to local cache!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }

  SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().first;
  SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().second;
  for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName() ) { break; } }
  if ( iter == dbParams_.partitions().second ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition \"" << partition
       << "\" not found in partition list, "
       << " therefore cannot add device descriptions!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  DeviceDescriptionsRange range = devices_.find( partition );
  if ( range == devices_.emptyRange() ) {
    
    // Make local copy 
    std::vector<DeviceDescription*> tmp;
#ifdef USING_NEW_DATABASE_MODEL
    FecFactory::vectorCopyI( tmp, devs, true );
#else
    tmp = devs;
#endif
    
    // Add to local cache
    devices_.loadNext( partition, tmp );

    // Some debug
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Added " << devs.size() 
       << " device descriptions to local cache for partition \""
       << partition << "\"."
       << " (Cache holds device descriptions for " 
       << devices_.size() << " partitions.)";
    LogTrace(mlConfigDb_) << ss.str();
    
  } else {
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Partition \"" << partition
       << "\" already found in local cache, "
       << " therefore cannot add device descriptions!";
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadDeviceDescriptions( std::string partition ) {

  if ( dbParams_.usingDbCache() ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]" 
      << " Using database cache! No uploads allowed!"; 
    return;
  }
  
  if ( !deviceFactory(__func__) ) { return; }
  
  if ( devices_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached device descriptions, therefore no upload!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  try { 

    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().first;
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().second;
    for ( ; iter != jter; ++iter ) {
      
      if ( partition == "" || partition == iter->second.partitionName() ) {
	
	DeviceDescriptionsRange range = devices_.find( iter->second.partitionName() );
	if ( range != devices_.emptyRange() ) {
	  
	  std::vector<DeviceDescription*> devs( range.begin(), range.end() );
	  
	  deviceFactory(__func__)->setFecDeviceDescriptions( devs,
							     iter->second.partitionName(),
							     &(iter->second.fecVersion().first),
							     &(iter->second.fecVersion().second),
							     true ); // new major version

	  // Some debug
	  std::stringstream ss;
	  ss << "[SiStripConfigDb::" << __func__ << "]"
	     << " Uploaded " << devs.size() 
	     << " device descriptions to database for partition \""
	     << iter->second.partitionName() << "\".";
	  LogTrace(mlConfigDb_) << ss.str();
	  
	} else {
	  stringstream ss; 
	  ss << "[SiStripConfigDb::" << __func__ << "]" 
	     << " Vector of device descriptions is empty for partition \"" 
	     << iter->second.partitionName()
	     << "\", therefore aborting upload for this partition!";
	  edm::LogWarning(mlConfigDb_) << ss.str(); 
	  continue; 
	}
	
      } else {
	// 	  stringstream ss; 
	// 	  ss << "[SiStripConfigDb::" << __func__ << "]" 
	// 	     << " Cannot find partition \"" << partition
	// 	     << "\" in cached partitions list: \""
	// 	     << dbParams_.partitionNames( dbParams_.partitionNames() ) 
	// 	     << "\", therefore aborting upload for this partition!";
	// 	  edm::LogWarning(mlConfigDb_) << ss.str(); 
      }
      
    }
    
  } catch (...) { handleException( __func__ ); }
  
  allowCalibUpload_ = true;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::clearDeviceDescriptions( std::string partition ) {
  LogTrace(mlConfigDb_) << "[SiStripConfigDb::" << __func__ << "]";
  
  if ( devices_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached device descriptions!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  // Reproduce temporary cache for "all partitions except specified one" (or clear all if none specified)
  DeviceDescriptions temporary_cache;
  if ( partition == ""  ) { temporary_cache = DeviceDescriptions(); }
  else {
    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().first;
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().second;
    for ( ; iter != jter; ++iter ) {
      if ( partition != iter->second.partitionName() ) {
	DeviceDescriptionsRange range = devices_.find( iter->second.partitionName() );
	if ( range != devices_.emptyRange() ) {
	  temporary_cache.loadNext( partition, std::vector<DeviceDescription*>( range.begin(), range.end() ) );
	} else {
	  // 	  stringstream ss; 
	  // 	  ss << "[SiStripConfigDb::" << __func__ << "]" 
	  // 	     << " Cannot find partition \"" << iter->second.partitionName()
	  // 	     << "\" in local cache!";
	  // 	  edm::LogWarning(mlConfigDb_) << ss.str(); 
	}
      }
    }
  }

  // Delete objects in local cache for specified partition (or all if not specified) 
  DeviceDescriptionsRange devs;
  if ( partition == "" ) { 
    if ( !devices_.empty() ) {
      devs = DeviceDescriptionsRange( devices_.find( dbParams_.partitions().first->second.partitionName() ).begin(),
					devices_.find( (--(dbParams_.partitions().second))->second.partitionName() ).end() );
    } else { devs = devices_.emptyRange(); }
  } else {
    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partitions().first;
    SiStripDbParams::SiStripPartitions::const_iterator jter = dbParams_.partitions().second;
    for ( ; iter != jter; ++iter ) { if ( partition == iter->second.partitionName() ) { break; } }
    devs = devices_.find( iter->second.partitionName() );
  }
  
  if ( devs != devices_.emptyRange() ) {

#ifdef USING_NEW_DATABASE_MODEL	
    std::vector<DeviceDescription*>::const_iterator ifed = devs.begin();
    std::vector<DeviceDescription*>::const_iterator jfed = devs.end();
    for ( ; ifed != jfed; ++ifed ) { if ( *ifed ) { delete *ifed; } }
#endif
    
  } else {
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]";
    if ( partition == "" ) { ss << " Found no device descriptions in local cache!"; }
    else { ss << " Found no device descriptions in local cache for partition \"" << partition << "\"!"; }
    edm::LogWarning(mlConfigDb_) << ss.str(); 
  }
  
  // Overwrite local cache with temporary cache
  devices_ = temporary_cache; 

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::printDeviceDescriptions( std::string partition ) {

  std::stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Contents of DeviceDescriptions container:" << std::endl;
  ss << " Number of partitions: " << devices_.size() << std::endl;

  // Loop through partitions
  uint16_t cntr = 0;
  DeviceDescriptions::const_iterator idev = devices_.begin();
  DeviceDescriptions::const_iterator jdev = devices_.end();
  for ( ; idev != jdev; ++idev ) {

    cntr++;
    if ( partition == "" || partition == idev->first ) {
      
      ss << "  Partition number : " << cntr << " (out of " << devices_.size() << ")" << std::endl;
      ss << "  Partition name   : " << idev->first << std::endl;
      ss << "  Num of devices   : " << idev->second.size() << std::endl;
      
      // Extract FEC crate, slot, etc
      std::map< uint32_t, vector<std::string> > devices;
      std::vector<DeviceDescription*>::const_iterator iter = idev->second.begin();
      std::vector<DeviceDescription*>::const_iterator jter = idev->second.end();
      for ( ; iter != jter; ++iter ) { 
	if ( *iter ) { 
	  DeviceAddress addr = deviceAddress( **iter );
	  uint32_t key  = SiStripFecKey( addr.fecCrate_, 
					 addr.fecSlot_, 
					 addr.fecRing_, 
					 0, 
					 0, 
					 0, 
					 0 ).key();
	  std::stringstream data;
	  data << (*iter)->getDeviceType() 
	       << "_"
	       << SiStripFecKey( addr.fecCrate_, 
				 addr.fecSlot_, 
				 addr.fecRing_, 
				 addr.ccuAddr_, 
				 addr.ccuChan_, 
				 addr.lldChan_, 
				 addr.i2cAddr_ ).key();
	  if ( find( devices[key].begin(), devices[key].end(), data.str() ) == devices[key].end() ) { 
	    devices[key].push_back( data.str() );
	  }
	}
      }
      
      // Sort contents
      std::map< uint32_t, std::vector<std::string> > tmp;
      std::map< uint32_t, std::vector<std::string> >::const_iterator ii = devices.begin();
      std::map< uint32_t, std::vector<std::string> >::const_iterator jj = devices.end();
      for ( ; ii != jj; ++ii ) {
	std::vector<std::string> temp = ii->second;
	std::sort( temp.begin(), temp.end() );
	std::vector<std::string>::const_iterator iii = temp.begin();
	std::vector<std::string>::const_iterator jjj = temp.end();
	for ( ; iii != jjj; ++iii ) { tmp[ii->first].push_back( *iii ); }
      }
      devices.clear();
      devices = tmp;
      
      // Print FEC crate, slot, etc...
      std::map< uint32_t, std::vector<std::string> >::const_iterator idev = devices.begin();
      std::map< uint32_t, std::vector<std::string> >::const_iterator jdev = devices.end();
      for ( ; idev != jdev; ++idev ) {
	SiStripFecKey key(idev->first);
	ss << "  Found " << std::setw(3) << idev->second.size()
	   << " devices for FEC crate/slot/ring " 
	   << key.fecCrate() << "/"
	   << key.fecSlot() << "/"
	   << key.fecRing();
	//<< " (ccu/module/lld/i2c): ";
	// 	if ( !idev->second.empty() ) { 
	// 	  uint16_t first = idev->second.front();
	// 	  uint16_t last = idev->second.front();
	// 	  std::vector<std::string>::const_iterator chan = idev->second.begin();
	// 	  for ( ; chan != idev->second.end(); chan++ ) { 
	// 	    if ( chan != idev->second.begin() ) {
	// 	      if ( *chan != last+1 ) { 
	// 		ss << std::setw(2) << first << "->" << std::setw(2) << last << ", ";
	// 		if ( chan != idev->second.end() ) { first = *(chan+1); }
	// 	      } 
	// 	    }
	// 	    last = *chan;
	// 	  }
	// 	  if ( first != last ) { ss << std::setw(2) << first << "->" << std::setw(2) << last; }
	ss << std::endl;
      }

    }
    
  }
  
  LogTrace(mlConfigDb_) << ss.str();

}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DeviceAddress SiStripConfigDb::deviceAddress( const deviceDescription& description ) {
  
  deviceDescription& desc = const_cast<deviceDescription&>(description); 
  
  DeviceAddress addr;
  try {
#ifdef USING_NEW_DATABASE_MODEL
    addr.fecCrate_ = static_cast<uint16_t>( desc.getCrateSlot() + sistrip::FEC_CRATE_OFFSET ); //@@ temporary offset?
#else
    addr.fecCrate_ = static_cast<uint16_t>( 0 + sistrip::FEC_CRATE_OFFSET ); //@@ temporary offset?
#endif
    addr.fecSlot_  = static_cast<uint16_t>( desc.getFecSlot() );
    addr.fecRing_  = static_cast<uint16_t>( desc.getRingSlot() + sistrip::FEC_RING_OFFSET ); //@@ temporary offset?
    addr.ccuAddr_  = static_cast<uint16_t>( desc.getCcuAddress() );
    addr.ccuChan_  = static_cast<uint16_t>( desc.getChannel() );
    addr.lldChan_  = static_cast<uint16_t>( SiStripFecKey::lldChan( desc.getAddress() ) );
    addr.i2cAddr_  = static_cast<uint16_t>( desc.getAddress() );
  } catch (...) { handleException( __func__ ); }
  
  return addr;
}

// -----------------------------------------------------------------------------
//
string SiStripConfigDb::deviceType( const enumDeviceType& device_type ) const {
  if      ( device_type == PLL )         { return "PLL"; }
  else if ( device_type == LASERDRIVER ) { return "LLD"; }
  else if ( device_type == DOH )         { return "DOH"; }
  else if ( device_type == APVMUX )      { return "MUX"; }
  else if ( device_type == APV25 )       { return "APV"; }
  else if ( device_type == DCU )         { return "DCU"; }
  else if ( device_type == GOH )         { return "GOH"; }
  else { return "UNKNOWN DEVICE!"; }
}
