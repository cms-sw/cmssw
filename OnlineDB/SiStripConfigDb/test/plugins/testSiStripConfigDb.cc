// Last commit: $Id: testSiStripConfigDb.cc,v 1.4 2008/04/22 12:40:57 bainbrid Exp $

#include "OnlineDB/SiStripConfigDb/test/plugins/testSiStripConfigDb.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
testSiStripConfigDb::testSiStripConfigDb( const edm::ParameterSet& pset ) 
  : db_(0),
    download_( pset.getUntrackedParameter<bool>("Download",false) ),
    upload_( pset.getUntrackedParameter<bool>("Upload",false) ),
    devices_( pset.getUntrackedParameter<bool>("DeviceDescriptions",false) ),
    feds_( pset.getUntrackedParameter<bool>("FedDescriptions",false) ),
    conns_( pset.getUntrackedParameter<bool>("FedConnections",false) ),
    dcus_( pset.getUntrackedParameter<bool>("DcuDetIdMap",false) )
{
  std::stringstream ss;
  ss << "[testSiStripConfigDb::" << __func__ << "]"
     << " Parameters:" << std::endl 
     << "  Download           : " << download_ << std::endl 
     << "  Upload             : " << upload_ << std::endl 
     << "  FedConnections     : " << conns_ << std::endl 
     << "  DeviceDescriptions : " << devices_ << std::endl 
     << "  FedDescriptions    : " << feds_ << std::endl 
     << "  DcuDetIdMap        : " << dcus_;
  LogTrace(mlCabling_) << ss.str();
}

// -----------------------------------------------------------------------------
// 
testSiStripConfigDb::~testSiStripConfigDb() {
  LogTrace(mlCabling_)
    << "[testSiStripConfigDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
void testSiStripConfigDb::beginJob( const edm::EventSetup& setup ) {

  // Access service
  db_ = edm::Service<SiStripConfigDb>().operator->();

  // Check pointer
  if ( db_ ) {
    edm::LogVerbatim(mlCabling_)
      << "[testSiStripConfigDb::" << __func__ << "]"
      << " Pointer to SiStripConfigDb: 0x" 
      << std::setw(8) << std::setfill('0')
      << std::hex << db_ << std::dec;
  } else {
    edm::LogError(mlCabling_)
      << "[testSiStripConfigDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb!"
      << " Aborting!";
    return;
  }

  // Check pointer
  if ( db_->deviceFactory() || db_->databaseCache() ) {
    std::stringstream ss;
    ss << "[testSiStripConfigDb::" << __func__ << "]" << std::endl
       << " Pointer to DeviceFactory: 0x" 
       << std::setw(8) << std::setfill('0') 
       << std::hex << db_->deviceFactory() << std::dec
       << std::endl
       << " Pointer to DatabaseCache: 0x" 
       << std::setw(8) << std::setfill('0') 
       << std::hex << db_->databaseCache() << std::dec
       << std::endl;
    edm::LogVerbatim(mlCabling_) << ss.str();
  } else {
    edm::LogError(mlCabling_)
      << "[testSiStripConfigDb::" << __func__ << "]"
      << " NULL pointer to DeviceFactory AND DatabaseCache!"
      << " Aborting!";
    return;
  }

  // Local caches
  std::stringstream ss;
  SiStripConfigDb::DeviceDescriptions devices;
  SiStripConfigDb::FedDescriptions feds;
  SiStripConfigDb::FedConnections conns;
  SiStripConfigDb::DcuDetIdMap dcus;

  // Downloads
  if ( download_ ) {
    
    // Connections
    if ( conns_ ) {

      // iterate through partitions and get, print, clear, print
      SiStripDbParams::SiStripPartitions::const_iterator iter = db_->dbParams().partitions_.begin();
      SiStripDbParams::SiStripPartitions::const_iterator jter = db_->dbParams().partitions_.end();
      for ( ; iter != jter; ++iter ) {
	SiStripConfigDb::FedConnections::range conns = db_->getFedConnections( iter->second.partitionName_ );
	db_->printFedConnections( iter->second.partitionName_ );
	db_->clearFedConnections( iter->second.partitionName_ );
	db_->printFedConnections( iter->second.partitionName_ );
	std::stringstream ss;
	ss << "[testSiStripConfigDb::" << __func__ << "]" 
	   << " Downloaded " << conns.size()
	   << " FED connections!";
	if ( !conns.empty() ) { edm::LogVerbatim("testSiStripConfigDb") << ss.str(); }
	else { edm::LogWarning("testSiStripConfigDb") << ss.str(); }
      }

      // get all partitions and print, clear, print
      SiStripConfigDb::FedConnections::range conns = db_->getFedConnections();
      db_->printFedConnections();
      db_->clearFedConnections();
      db_->printFedConnections();
      std::stringstream ss;
      ss << "[testSiStripConfigDb::" << __func__ << "]" 
	 << " Downloaded " << conns.size()
	 << " FED connections!";
      if ( !conns.empty() ) { edm::LogVerbatim("testSiStripConfigDb") << ss.str(); }
      else { edm::LogWarning("testSiStripConfigDb") << ss.str(); }

    }

    // Devices
    if ( devices_ ) {
      devices = db_->getDeviceDescriptions();
      std::stringstream ss;
      ss << "[testSiStripConfigDb::" << __func__ << "]" 
	 << " Downloaded " << devices.size() 
	 << " device descriptions!";
      if ( !devices.empty() ) { edm::LogVerbatim("testSiStripConfigDb") << ss.str(); }
      else { edm::LogWarning("testSiStripConfigDb") << ss.str(); }
    }

    // FEDs
    if ( feds_ ) {
      feds = db_->getFedDescriptions();
      std::stringstream ss;
      ss << "[testSiStripConfigDb::" << __func__ << "]" 
	 << " Downloaded " << feds.size() 
	 << " FED descriptions!";
      if ( !feds.empty() ) { edm::LogVerbatim("testSiStripConfigDb") << ss.str(); }
      else { edm::LogWarning("testSiStripConfigDb") << ss.str(); }
    }

    // DCU-DetId map
    if ( dcus_ ) {
      dcus = db_->getDcuDetIdMap();
      std::stringstream ss;
      ss << "[testSiStripConfigDb::" << __func__ << "]" 
	 << " Downloaded " << dcus.size() 
	 << " entries in DCU-DetId map!";
      if ( !dcus.empty() ) { edm::LogVerbatim("testSiStripConfigDb") << ss.str(); }
      else { edm::LogWarning("testSiStripConfigDb") << ss.str(); }
    }
    
  }

  // Uploads
  if ( upload_ ) {

    // Connections
    if ( conns_ ) {

      // build temporary cache and print, clear (local cache)
      SiStripConfigDb::FedConnections connections;
      SiStripDbParams::SiStripPartitions::const_iterator ii = db_->dbParams().partitions_.begin();
      SiStripDbParams::SiStripPartitions::const_iterator jj = db_->dbParams().partitions_.end();
      for ( ; ii != jj; ++ii ) {
	SiStripConfigDb::FedConnections::range conns = db_->getFedConnections( ii->second.partitionName_ );
	if ( conns != connections.emptyRange() ) {
	  std::vector<SiStripConfigDb::FedConnection*> tmp1( conns.begin(), conns.end() );
	  std::vector<SiStripConfigDb::FedConnection*> tmp2;
#ifdef USING_NEW_DATABASE_MODEL
	  ConnectionFactory::vectorCopyI( tmp2, tmp1, true );
#else
	  tmp2 = tmp1;
#endif
	  connections.loadNext( ii->second.partitionName_, tmp2 );
	}
      }
      db_->printFedConnections();
      db_->clearFedConnections();

      // iterate through partitions and add, print and upload
      SiStripDbParams::SiStripPartitions::const_iterator iter = db_->dbParams().partitions_.begin();
      SiStripDbParams::SiStripPartitions::const_iterator jter = db_->dbParams().partitions_.end();
      for ( ; iter != jter; ++iter ) {
	SiStripConfigDb::FedConnections::range conns = connections.find( iter->second.partitionName_ );
	std::vector<SiStripConfigDb::FedConnection*> temp( conns.begin(), conns.end() );
	db_->addFedConnections( iter->second.partitionName_, temp );
	db_->printFedConnections( iter->second.partitionName_ );
	db_->uploadFedConnections( iter->second.partitionName_ );
      }

      // print all partitions and then upload, clear, print
      db_->printFedConnections();
      db_->uploadFedConnections();
      db_->clearFedConnections();
      db_->printFedConnections();
      
    }

  }
  
}


