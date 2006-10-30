#include "OnlineDB/SiStripESSources/test/stubs/test_FedCablingBuilder.h"
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
//
#include <iostream>
#include <sstream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
test_FedCablingBuilder::test_FedCablingBuilder( const edm::ParameterSet& pset ) 
  : db_(0),
    source_( pset.getUntrackedParameter<string>( "Source", "CONNECTIONS" ) )
{
  LogDebug(mlCabling_)
    << "[test_FedCablingBuilder::" << __func__ << "]"
    << " Constructing object...";
  
  if ( pset.getUntrackedParameter<bool>( "UsingDb", true ) ) {
    db_ = new SiStripConfigDb( pset.getUntrackedParameter<string>("ConfDb",""),
			       pset.getUntrackedParameter<string>("Partition",""),
			       pset.getUntrackedParameter<unsigned int>("MajorVersion",0),
			       pset.getUntrackedParameter<unsigned int>("MinorVersion",0) );
    if ( db_ ) { db_->openDbConnection(); }
  } else {
    edm::LogError(mlCabling_)
      << "[test_FedCablingBuilder::" << __func__ << "]"
      << " Cannot use database! 'UsingDb' is false!";
  }
  
  LogDebug(mlCabling_)
    << "[test_FedCablingBuilder::" << __func__ << "]"
    << " 'SOURCE' configurable set to: " << source_;
  
}

// -----------------------------------------------------------------------------
// 
test_FedCablingBuilder::~test_FedCablingBuilder() {
  if ( db_ ) { 
    db_->closeDbConnection();
    delete db_;
  } 
  LogDebug(mlCabling_)
    << "[test_FedCablingBuilder::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
void test_FedCablingBuilder::beginJob( const edm::EventSetup& setup ) {
  
  SiStripFecCabling fec_cabling;
  SiStripConfigDb::DcuDetIdMap dcu_detid_map;
  
  // Build FED cabling
  if ( source_ == "CONNECTIONS" ) { 
    SiStripFedCablingBuilderFromDb::buildFecCablingFromFedConnections( db_, 
								       fec_cabling, 
								       dcu_detid_map );
  } else if ( source_ == "DEVICES" ) {
    SiStripFedCablingBuilderFromDb::buildFecCablingFromDevices( db_, 
								fec_cabling, 
								dcu_detid_map );
  } else if ( source_ == "DETID" ) { 
    SiStripFedCablingBuilderFromDb::buildFecCablingFromDetIds( db_, 
							       fec_cabling, 
							       dcu_detid_map );
  } else if ( source_ == "UNDEFINED" ) { 
    SiStripFedCablingBuilderFromDb::buildFecCabling( db_, 
						     fec_cabling, 
						     dcu_detid_map );
  } else { 
    edm::LogError(mlCabling_)
      << "[test_FedCablingBuilder::" << __func__ << "]"
      << " Unable to build FEC cabling!"
      << " Unexpected value for 'SOURCE' configurable: " << source_ << endl;
    return;
  }

  // Build FED cabling object
  SiStripFedCabling fed_cabling;
  SiStripFedCablingBuilderFromDb::getFedCabling( fec_cabling, fed_cabling );
  
  // Some debug
  const NumberOfDevices& devs = fec_cabling.countDevices();
  stringstream ss;
  ss << "[test_FedCablingBuilder::" << __func__ << "]"
     << " Built SiStripFecCabling object with following devices:" << endl
     << endl << devs;
  LogDebug(mlCabling_) << ss.str();

  stringstream sss;
  sss << "[test_FedCablingBuilder::" << __func__ << "]" << endl;
  sss << fed_cabling;
  LogDebug(mlCabling_) << sss.str();
  
}

