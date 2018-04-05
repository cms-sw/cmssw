#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "CalibFormats/SiStripObjects/interface/NumberOfDevices.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::CommissioningHistosUsingDb( SiStripConfigDb* const db,
                                                        sistrip::RunType type )
  : CommissioningHistograms(),
    runType_(type),
    db_(db),
    cabling_(nullptr),
    detInfo_(),
    uploadAnal_(true),
    uploadConf_(false)
{
  LogTrace(mlDqmClient_) 
    << "[" << __PRETTY_FUNCTION__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::CommissioningHistosUsingDb()
  : CommissioningHistograms(),
    runType_(sistrip::UNDEFINED_RUN_TYPE),
    db_(nullptr),
    cabling_(nullptr),
    detInfo_(),
    uploadAnal_(false),
    uploadConf_(false)
{
  LogTrace(mlDqmClient_) 
    << "[" << __PRETTY_FUNCTION__ << "]"
    << " Constructing object..." << endl;
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::~CommissioningHistosUsingDb() {
  if ( db_ ) { delete db_; }
  LogTrace(mlDqmClient_) 
    << "[" << __PRETTY_FUNCTION__ << "]"
    << " Destructing object...";
}

void CommissioningHistosUsingDb::configure( const edm::ParameterSet&, const edm::EventSetup& setup )
{
  if ( !db_ ) {
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Cannot configure...";
  } else {
    // Build FEC cabling object from connections found in DB
    SiStripFecCabling fec_cabling;
    if ( runType_ == sistrip::FAST_CABLING ) {
      SiStripFedCablingBuilderFromDb::buildFecCablingFromDevices( db_, fec_cabling );
    } else {
      SiStripFedCablingBuilderFromDb::buildFecCabling( db_, fec_cabling );
    }

    // Build FED cabling from FEC cabling
    cabling_ = new SiStripFedCabling();
    SiStripFedCablingBuilderFromDb::getFedCabling( fec_cabling, *cabling_ );
    std::stringstream ss;
    ss << "[CommissioningHistosUsingDb::" << __func__ << "]"
       << " Terse print out of FED cabling:" << std::endl;
    cabling_->terse(ss);
    LogTrace(mlDqmClient_) << ss.str();

    edm::ESHandle<TrackerTopology> tTopo;
    setup.get<TrackerTopologyRcd>().get(tTopo);
    std::stringstream sss;
    sss << "[CommissioningHistosUsingDb::" << __func__ << "]"
        << " Summary of FED cabling:" << std::endl;
    cabling_->summary(sss, tTopo.product());
    edm::LogVerbatim(mlDqmClient_) << sss.str();
  }
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistosUsingDb::uploadToConfigDb() {
  buildDetInfo();
  addDcuDetIds(); 
  uploadConfigurations();
  uploadAnalyses(); 
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistosUsingDb::uploadAnalyses() {

  if ( !db_ ) {
    edm::LogError(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
  db_->clearAnalysisDescriptions();
  SiStripDbParams::SiStripPartitions::const_iterator ip = db_->dbParams().partitions().begin();
  SiStripDbParams::SiStripPartitions::const_iterator jp = db_->dbParams().partitions().end();
  for ( ; ip != jp; ++ip ) {

    edm::LogVerbatim(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " Starting from partition " << ip->first
      << " with versions:\n" << std::dec
      << "   Conn: " << ip->second.cabVersion().first << "." << ip->second.cabVersion().second << "\n"
      << "   FED:  " << ip->second.fedVersion().first  << "." << ip->second.fedVersion().second << "\n"
      << "   FEC:  " << ip->second.fecVersion().first  << "." << ip->second.fecVersion().second << "\n"
      << "   Mask: " << ip->second.maskVersion().first << "." << ip->second.maskVersion().second;

    // Upload commissioning analysis results 
    SiStripConfigDb::AnalysisDescriptionsV anals;
    createAnalyses( anals );
    
    edm::LogVerbatim(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " Created analysis descriptions for " 
      << anals.size() << " devices";
    
    // Update analysis descriptions with new commissioning results
    if ( uploadAnal_ ) {
      if ( uploadConf_ ) { 
				edm::LogVerbatim(mlDqmClient_)
	  		<< "[CommissioningHistosUsingDb::" << __func__ << "]"
	  		<< " Uploading major version of analysis descriptions to DB"
	  		<< " (will be used for physics)...";
      } 
      else {
				edm::LogVerbatim(mlDqmClient_)
	 			<< "[CommissioningHistosUsingDb::" << __func__ << "]"
	  		<< " Uploading minor version of analysis descriptions to DB"
	  		<< " (will not be used for physics)...";
      }
      db_->clearAnalysisDescriptions( ip->second.partitionName() );
      db_->addAnalysisDescriptions( ip->second.partitionName(), anals ); 
      db_->uploadAnalysisDescriptions( uploadConf_, ip->second.partitionName() ); 
      edm::LogVerbatim(mlDqmClient_) 
			<< "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " Upload of analysis descriptions to DB finished!";
    } 
    else {
      edm::LogWarning(mlDqmClient_) 
        << "[CommissioningHistosUsingDb::" << __func__ << "]"
        << " TEST! No analysis descriptions will be uploaded to DB...";
    }

    if ( uploadConf_ ) {
      SiStripDbParams::SiStripPartitions::const_iterator ip = db_->dbParams().partitions().begin();
      SiStripDbParams::SiStripPartitions::const_iterator jp = db_->dbParams().partitions().end();
      for ( ; ip != jp; ++ip ) {
        DeviceFactory* df = db_->deviceFactory();
        tkStateVector states = df->getCurrentStates();
        tkStateVector::const_iterator istate = states.begin();
        tkStateVector::const_iterator jstate = states.end();
        while ( istate != jstate ) {
          if ( *istate && ip->first == (*istate)->getPartitionName() ) { break; }
          istate++;
        }
        // Set versions if state was found
        if ( istate != states.end() ) {
          edm::LogVerbatim(mlDqmClient_) 
            << "[CommissioningHistosUsingDb::" << __func__ << "]"
            << " Created new version for partition " << ip->first
            << ". Current state:\n" << std::dec
            << "   Conn: " << (*istate)->getConnectionVersionMajorId() << "." << (*istate)->getConnectionVersionMinorId() << "\n"
            << "   FED:  " << (*istate)->getFedVersionMajorId()  << "." << (*istate)->getFedVersionMinorId() << "\n"
            << "   FEC:  " << (*istate)->getFecVersionMajorId()  << "." << (*istate)->getFecVersionMinorId() << "\n"
            << "   Mask: " << (*istate)->getMaskVersionMajorId() << "." << (*istate)->getMaskVersionMinorId();
        }
      }
    }

  }
  
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistosUsingDb::addDcuDetIds() {
  
  if ( !cabling_ ) {
    edm::LogWarning(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripFedCabling object!";
    return;
  }
  
  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for ( ; ianal != janal; ++ianal ) { 

    CommissioningAnalysis* anal = ianal->second;
  
    if ( !anal ) {
      edm::LogWarning(mlDqmClient_) 
	<< "[CommissioningHistosUsingDb::" << __func__ << "]"
	<< " NULL pointer to CommissioningAnalysis object!";
      return;
    }
    
    SiStripFedKey fed_key = anal->fedKey();
    SiStripFecKey fec_key = anal->fecKey();
    
    FedChannelConnection conn = cabling_->fedConnection( fed_key.fedId(),
                                                         fed_key.fedChannel() );
  
    SiStripFedKey fed( conn.fedId(),
		       SiStripFedKey::feUnit( conn.fedCh() ),
		       SiStripFedKey::feChan( conn.fedCh() ) );
  
    SiStripFecKey fec( conn.fecCrate(),
		       conn.fecSlot(),
		       conn.fecRing(),
		       conn.ccuAddr(),
		       conn.ccuChan(),
		       conn.lldChannel() );
  
    if ( fed_key.path() != fed.path() ) {

      std::stringstream ss;
      ss << "[CommissioningHistosUsingDb::" << __func__ << "]"
	 << " Cannot set DCU and DetId values in commissioning analysis object!" << std::endl
	 << " Incompatible FED key retrieved from cabling!" << std::endl
	 << " FED key from analysis object  : " << fed_key.path() << std::endl
	 << " FED key from cabling object   : " << fed.path() << std::endl
	 << " FED id/ch from analysis object: " << fed_key.fedId() << "/" << fed_key.fedChannel() << std::endl
	 << " FED id/ch from cabling object : " << conn.fedId() << "/" << conn.fedCh();
      edm::LogWarning(mlDqmClient_) << ss.str();

    } else if ( fec_key.path() != fec.path() ) {

      std::stringstream ss;
      ss << "[CommissioningHistosUsingDb::" << __func__ << "]"
	 << " Cannot set DCU and DetId values in commissioning analysis object!" << std::endl
	 << " Incompatible FEC key retrieved from cabling!" << std::endl
	 << " FEC key from analysis object : " << fec_key.path() << std::endl
	 << " FEC key from cabling object  : " << fec.path();
      edm::LogWarning(mlDqmClient_) << ss.str();

    } else {

      anal->dcuId( conn.dcuId() );
      anal->detId( conn.detId() );

    }

  }

}

// -----------------------------------------------------------------------------
//
void CommissioningHistosUsingDb::createAnalyses( SiStripConfigDb::AnalysisDescriptionsV& desc ) {
  
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistosUsingDb::" << __func__ << "]"
    << " Creating AnalysisDescriptions...";

  desc.clear();
  
//   uint16_t size = 0;
//   std::stringstream ss;
//   ss << "[CommissioningHistosUsingDb::" << __func__ << "]"
//      << " Analysis descriptions:" << std::endl;

  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for ( ; ianal != janal; ++ianal ) { 

    // create analysis description
    create( desc, ianal ); 
    
//     // debug
//     if ( ianal->second ) {
//       if ( desc.size()/2 > size ) { // print every 2nd description
// 	size = desc.size()/2;
// 	ianal->second->print(ss); 
// 	ss << (*(desc.end()-2))->toString();
// 	ss << (*(desc.end()-1))->toString();
// 	ss << std::endl;
//       }
//     }

  }

//   LogTrace(mlDqmClient_) << ss.str(); 
  
}

// -----------------------------------------------------------------------------
//
void CommissioningHistosUsingDb::buildDetInfo() {
  
  detInfo_.clear();
  
  if ( !db() ) {
    edm::LogError(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!";
    return;
  }

  SiStripDbParams::SiStripPartitions::const_iterator ii = db_->dbParams().partitions().begin();
  SiStripDbParams::SiStripPartitions::const_iterator jj = db_->dbParams().partitions().end();
  for ( ; ii != jj; ++ii ) {
    
    // Retrieve DCUs and DetIds for given partition
    std::string pp = ii->second.partitionName();
    SiStripConfigDb::DeviceDescriptionsRange dcus = db()->getDeviceDescriptions( DCU, pp ); 
    SiStripConfigDb::DcuDetIdsRange dets = db()->getDcuDetIds( pp ); 
    
    // Iterate through DCUs
    SiStripConfigDb::DeviceDescriptionsV::const_iterator idcu = dcus.begin();
    SiStripConfigDb::DeviceDescriptionsV::const_iterator jdcu = dcus.end();
    for ( ; idcu != jdcu; ++idcu ) {
      
      // Extract DCU-FEH description 
      dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idcu );
      if ( !dcu ) { continue; }
      if ( dcu->getDcuType() != "FEH" ) { continue; }

      // S.L. 29/1/2010
      // HARDCODED!!! We have a broken module, known from Pisa integration tests
      // We could really use a better solutin for this than hardcode it!!!
      if ( dcu->getDcuHardId() == 16448250 ) continue; // fake dcu (0xfafafa)

      // Find TkDcuInfo object corresponding to given DCU description
      SiStripConfigDb::DcuDetIdsV::const_iterator idet = dets.end();
      idet = SiStripConfigDb::findDcuDetId( dets.begin(), dets.end(), dcu->getDcuHardId() );
      if ( idet == dets.begin() ) { continue; }
      
      // Extract TkDcuInfo object
      TkDcuInfo* det = idet->second;
      if ( !det ) { continue; }
      
      // Build FEC key
      const SiStripConfigDb::DeviceAddress& addr = db()->deviceAddress( *dcu );
      SiStripFecKey fec_key( addr.fecCrate_,
			     addr.fecSlot_,
			     addr.fecRing_,
			     addr.ccuAddr_,
			     addr.ccuChan_ );
      
      // Build DetInfo object
      DetInfo info;
      info.dcuId_ = det->getDcuHardId();
      info.detId_ = det->getDetId();
      info.pairs_ = det->getApvNumber()/2; 

      // Add it to map
      if ( fec_key.isValid() ) { detInfo_[pp][fec_key.key()] = info; }
      
    } 
  }
  
  // Debug
  if ( edm::isDebugEnabled() ) {
    std::stringstream ss;
    ss << "[CommissioningHistosUsingDb::" << __func__ << "]"
       << " List of modules for "
       << detInfo_.size()
       << " partitions, with their DCUids, DetIds, and nApvPairs: " << std::endl;
    std::map<std::string,DetInfos>::const_iterator ii = detInfo_.begin();
    std::map<std::string,DetInfos>::const_iterator jj = detInfo_.end();
    for ( ; ii != jj; ++ii ) {
      ss << " Partition \"" << ii->first 
	 << "\" has " << ii->second.size()
	 << " modules:"
	 << std::endl;
      DetInfos::const_iterator iii = ii->second.begin();
      DetInfos::const_iterator jjj = ii->second.end();
      for ( ; iii != jjj; ++iii ) {
	SiStripFecKey key = iii->first;
	ss << "  module= "
	   << key.fecCrate() << "/"    
	   << key.fecSlot() << "/"
	   << key.fecRing() << "/"
	   << key.ccuAddr() << "/"
	   << key.ccuChan() << ", "
	   << std::hex
	   << " DCUid= " 
	   << std::setw(8) << std::setfill('0') << iii->second.dcuId_
	   << " DetId= " 
	   << std::setw(8) << std::setfill('0') << iii->second.detId_
	   << std::dec
	   << " nPairs= "
	   << iii->second.pairs_
	   << std::endl;
      }
    }
    //LogTrace(mlDqmClient_) << ss.str();
  }
  
}

// -----------------------------------------------------------------------------
//
std::pair<std::string,CommissioningHistosUsingDb::DetInfo> CommissioningHistosUsingDb::detInfo( const SiStripFecKey& key ) {
  SiStripFecKey tmp( key, sistrip::CCU_CHAN );
  if ( tmp.isInvalid() ) { return std::make_pair("",DetInfo()); }
  std::map<std::string,DetInfos>::const_iterator ii = detInfo_.begin();
  std::map<std::string,DetInfos>::const_iterator jj = detInfo_.end();
  for ( ; ii != jj; ++ii ) {
    DetInfos::const_iterator iii = ii->second.find( tmp.key() );
    if ( iii != ii->second.end() ) { return std::make_pair(ii->first,iii->second); }
  }
  return std::make_pair("",DetInfo());
}

// -----------------------------------------------------------------------------
//
bool CommissioningHistosUsingDb::deviceIsPresent( const SiStripFecKey& key ) {
  SiStripFecKey tmp( key, sistrip::CCU_CHAN );
  std::pair<std::string,DetInfo> info = detInfo(key);
  if ( info.second.dcuId_ != sistrip::invalid32_ ) {
    if ( key.channel() == 2 && info.second.pairs_ == 2 ) { return false; }
    else { return true; }
  } else {
    std::stringstream ss;
    ss << "[CommissioningHistosUsingDb::" << __func__ << "]"
       << " Cannot find module (crate/FEC/ring/CCU/module): "
       << tmp.fecCrate() << "/"
       << tmp.fecSlot() << "/"
       << tmp.fecRing() << "/"
       << tmp.ccuAddr() << "/"
       << tmp.ccuChan()
       << "!";
    edm::LogWarning(mlDqmClient_) << ss.str();
    return true;
  }
}



  
