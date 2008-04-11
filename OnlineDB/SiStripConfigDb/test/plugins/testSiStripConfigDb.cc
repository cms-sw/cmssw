// Last commit: $Id: testSiStripConfigDb.cc,v 1.1 2008/03/26 09:13:11 bainbrid Exp $

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
     << "  DeviceDescriptions : " << devices_ << std::endl 
     << "  FedDescriptions    : " << feds_ << std::endl 
     << "  FedConnections     : " << conns_ << std::endl 
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

    // Connections
    if ( conns_ ) {
      SiStripConfigDb::FedConnections::range conns = db_->getFedConnections();
      std::stringstream ss;
      ss << "[testSiStripConfigDb::" << __func__ << "]" 
	 << " Downloaded " << conns.size()
	 << " FED connections!";
      if ( !conns.empty() ) { edm::LogVerbatim("testSiStripConfigDb") << ss.str(); }
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
  }
  
}

