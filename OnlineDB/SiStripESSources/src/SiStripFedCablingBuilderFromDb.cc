// Last commit: $Id$
// Latest tag:  $Name$
// Location:    $Source$

#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilder.h"

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderFromDb::SiStripFedCablingBuilderFromDb( const edm::ParameterSet& pset ) 
  : SiStripFedCablingESSource( pset ),
    db_(0),
    partitions_( pset.getUntrackedParameter< vector<string> >( "Partitions", vector<string>() ) )
{
  edm::LogInfo ("FedCabling") << "[SiStripFedCablingBuilderFromDb::SiStripFedCablingBuilderFromDb]"
			      << " Constructing object...";
  if ( pset.getUntrackedParameter<bool>( "UseConfigDb", true ) ) {
    // Using database 
    db_ = new SiStripConfigDb( pset.getUntrackedParameter<string>("User",""),
			       pset.getUntrackedParameter<string>("Passwd",""),
			       pset.getUntrackedParameter<string>("Path","") );
  } else {
    // Using xml files
    db_ = new SiStripConfigDb( pset.getUntrackedParameter<string>("ModuleXmlFile",""),
			       pset.getUntrackedParameter<string>("DcuInfoXmlFile",""),
			       pset.getUntrackedParameter< vector<string> >( "FecXmlFiles", vector<string>() ),
			       pset.getUntrackedParameter< vector<string> >( "FedXmlFiles", vector<string>() ) );
  }
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderFromDb::~SiStripFedCablingBuilderFromDb() {
  edm::LogInfo("FedCabling") << "[SiStripFedCablingBuilderFromDb::~SiStripFedCablingBuilderFromDb]"
			     << " Destructing object...";
  if ( db_ ) { 
    db_->closeDbConnection();
    delete db_; 
  } 
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCabling* SiStripFedCablingBuilderFromDb::makeFedCabling() {
  edm::LogInfo("FedCabling") << "[SiStripFedCablingBuilderFromDb::makeFedCabling]"
			     << " Building FED cabling...";
  
  // Attempt to connect to database
  if ( !db_->openDbConnection() ) { return 0; }
  
  // Create FED cabling object 
  SiStripFedCabling* fed_cabling = new SiStripFedCabling();

  // Create DcuId-DetId map
  SiStripConfigDb::DcuIdDetIdMap dcuid_detid_map;
  
  // Populate FEC cabling object
  if ( db_->buildFecCablingFromFecDevices() ) { 
    SiStripFedCablingBuilder::createFedCablingFromDevices( db_, 
							   *fed_cabling, 
							   dcuid_detid_map ); 
  } else { 
    SiStripFedCablingBuilder::createFedCablingFromFedConnections( db_, 
								  *fed_cabling,
								  dcuid_detid_map ); 
  }

  // Call virtual method that writes FED cabling object to conditions DB
  writeFedCablingToCondDb();
  
  return fed_cabling;
  
}








  
