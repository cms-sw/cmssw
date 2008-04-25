// Last commit: $Id: AnalysisDescriptions.cc,v 1.5 2008/04/21 09:33:01 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/AnalysisDescriptions.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#ifdef USING_NEW_DATABASE_MODEL

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::AnalysisDescriptions& SiStripConfigDb::getAnalysisDescriptions( const AnalysisType& analysis_type ) {
  
  analyses_.clear();
  if ( !deviceFactory(__func__) ) { return analyses_; }
  
  if ( dbParams_.partitions_.begin()->second.runType() == sistrip::PHYSICS ) { 
    
    // if physics run, return calibration constants 
    try { 

      //@@ GET LATEST RUN NUMBER HERE!!!!

      analyses_ = deviceFactory(__func__)->getCalibrationData( dbParams_.partitions_.begin()->second.runNumber(), 
							       dbParams_.partitions_.begin()->second.partitionName(), 
							       analysis_type );
    } catch (...) { handleException( __func__ ); }
    
  } else if ( dbParams_.partitions_.begin()->second.runType() != sistrip::PHYSICS &&
	      dbParams_.partitions_.begin()->second.runType() != sistrip::UNDEFINED_RUN_TYPE &&
	      dbParams_.partitions_.begin()->second.runType() != sistrip::UNKNOWN_RUN_TYPE ) { 
    
    // else if commissioning run, return version of analysis objects from "history" 

    // check if run type is consistent with analysis type requested
    if ( dbParams_.partitions_.begin()->second.runType() == sistrip::PEDESTALS      && analysis_type == AnalysisDescription::T_ANALYSIS_PEDESTALS ||
	 dbParams_.partitions_.begin()->second.runType() == sistrip::CALIBRATION    && analysis_type == AnalysisDescription::T_ANALYSIS_CALIBRATION ||
	 dbParams_.partitions_.begin()->second.runType() == sistrip::APV_LATENCY    && analysis_type == AnalysisDescription::T_ANALYSIS_APVLATENCY ||
	 dbParams_.partitions_.begin()->second.runType() == sistrip::FAST_CABLING   && analysis_type == AnalysisDescription::T_ANALYSIS_FASTFEDCABLING ||
	 dbParams_.partitions_.begin()->second.runType() == sistrip::FINE_DELAY_TTC && analysis_type == AnalysisDescription::T_ANALYSIS_FINEDELAY ||
	 dbParams_.partitions_.begin()->second.runType() == sistrip::OPTO_SCAN      && analysis_type == AnalysisDescription::T_ANALYSIS_OPTOSCAN ||
	 dbParams_.partitions_.begin()->second.runType() == sistrip::APV_TIMING     && analysis_type == AnalysisDescription::T_ANALYSIS_TIMING ||
	 dbParams_.partitions_.begin()->second.runType() == sistrip::VPSP_SCAN      && analysis_type == AnalysisDescription::T_ANALYSIS_VPSPSCAN ) {
      
      typedef std::pair<uint32_t,uint32_t> Version;
      typedef std::vector<Version> Versions;
      typedef std::map<uint32_t,Versions> Runs;
      
      // retrieve "history" first
      Runs runs;
      try { 
	runs = deviceFactory(__func__)->getAnalysisHistory( dbParams_.partitions_.begin()->second.partitionName(), 
							    analysis_type );
      } catch (...) { handleException( __func__ ); }

      // then retrieve appropriate version from "history"
      uint32_t major = 0;
      uint32_t minor = 0;
      if ( !runs.empty() ) {
	
	Runs::const_iterator irun = runs.end();
	if ( dbParams_.partitions_.begin()->second.runNumber() == 0 ) { irun = --(runs.end()); } //@@ assumes map is sorted
	else { irun = runs.find( dbParams_.partitions_.begin()->second.runNumber() ); } 
	
	if ( irun != runs.end() ) {
	  
	  // Build temp vector of "versions"
	  std::vector<uint16_t> vers;
	  if ( !irun->second.empty() ) {
	    Versions::const_iterator ivers = irun->second.begin();
	    Versions::const_iterator jvers = irun->second.end();
	    for ( ; ivers != jvers; ++ivers ) { 
	      vers.push_back( ivers->first * 1000000 + ivers->second );
	    }
	  }
	  
	  // Extract major / minor versions
	  if ( dbParams_.partitions_.begin()->second.calVersion().first == 0 && 
	       dbParams_.partitions_.begin()->second.calVersion().second == 0 ) { 
	    sort( vers.begin(), vers.end() ); 
	    major = vers.back() / 1000000;
	    minor = vers.back() % 1000000;
	  } else {
	    uint16_t key = 
	      dbParams_.partitions_.begin()->second.calVersion().first * 1000000 + 
	      dbParams_.partitions_.begin()->second.calVersion().second;
	    if ( find( vers.begin(), vers.end(), key ) != vers.end() ) {
	      major = dbParams_.partitions_.begin()->second.calVersion().first;
	      minor = dbParams_.partitions_.begin()->second.calVersion().second;
	    }
	  }

	} else {
	  edm::LogWarning(mlConfigDb_)
	    << "[SiStripConfigDb::" << __func__ << "]"
	    << " Could not find run " << dbParams_.partitions_.begin()->second.runNumber()
	    << " in history list!";
	}

      } else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " History list is empty! No runs found!";
      }

      // Retrieve analysis for given versions
      if ( major && minor ) {

	try { 
	  analyses_ = deviceFactory(__func__)->getAnalysisHistory( dbParams_.partitions_.begin()->second.partitionName(), 
								   major,
								   minor,
								   analysis_type );
	} catch (...) { handleException( __func__ ); }

	stringstream ss; 
	ss << "[SiStripConfigDb::" << __func__ << "]"
	   << " Found " << analyses_.size()
	   << " analysis descriptions (for analyses of type " 
	   << analysisType( analysis_type ) << ")"; 
	if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.partitions_.begin()->second.inputFecXml().size() << " 'fec.xml' file(s)"; }
	else { ss << " in database partition '" << dbParams_.partitions_.begin()->second.partitionName() << "'"; }
	if ( analyses_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
	else { LogTrace(mlConfigDb_) << ss.str(); }
	
	return analyses_;

      } else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " Cannot retrieve analysis objects for run number " << dbParams_.partitions_.begin()->second.runNumber()
	  << " run type " << SiStripEnumsAndStrings::runType( dbParams_.partitions_.begin()->second.runType() )
	  << " and version " << dbParams_.partitions_.begin()->second.calVersion().first << "." << dbParams_.partitions_.begin()->second.calVersion().second << "!";
      }
      
    } else {
      edm::LogWarning(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Run type \"" << SiStripEnumsAndStrings::runType( dbParams_.partitions_.begin()->second.runType() ) 
	<< " is not compatible with requested analysis type " 
	<< analysisType( analysis_type ) << "!";
    }
    
  } else {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Cannot retrieve analysis objects for run type " 
      << SiStripEnumsAndStrings::runType( dbParams_.partitions_.begin()->second.runType() );
  }
  
  return analyses_;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadAnalysisDescriptions( bool use_as_calibrations_for_physics ) {
  
  if ( !deviceFactory(__func__) ) { return; }
  
  if ( analyses_.empty() ) { 
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]" 
      << " Found no cached analysis descriptions, therefore no upload!"; 
    return; 
  }

  AnalysisType analysis_type = AnalysisDescription::T_UNKNOWN;
  if ( analyses_.front() ) { analysis_type = analyses_.front()->getType(); }
  if ( analysis_type == AnalysisDescription::T_UNKNOWN ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Analysis type is UNKNOWN. Aborting upload!";
    return;
  }

  if ( use_as_calibrations_for_physics && !allowCalibUpload_ ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Attempting to upload calibration constants without uploading any hardware descriptions!"
      << " Not allowed! Aborting upload!";
    return;
  } else { allowCalibUpload_ = false; }
  
  try { 
    uint32_t version = deviceFactory(__func__)->uploadAnalysis( dbParams_.partitions_.begin()->second.runNumber(), 
								dbParams_.partitions_.begin()->second.partitionName(), 
								analysis_type,
								analyses_,
								use_as_calibrations_for_physics );
    if ( use_as_calibrations_for_physics ) {
      deviceFactory(__func__)->uploadAnalysisState( version );
    }
  } catch (...) { handleException( __func__ ); }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::createAnalysisDescriptions( AnalysisDescriptions& desc ) {
  analyses_.clear();
  analyses_ = desc;
}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DeviceAddress SiStripConfigDb::deviceAddress( const AnalysisDescription& desc ) {
  
  DeviceAddress addr;
  try {
    addr.fecCrate_ = static_cast<uint16_t>( desc.getCrate() + sistrip::FEC_CRATE_OFFSET );
    addr.fecSlot_  = static_cast<uint16_t>( desc.getSlot() );
    addr.fecRing_  = static_cast<uint16_t>( desc.getRing() + sistrip::FEC_RING_OFFSET );
    addr.ccuAddr_  = static_cast<uint16_t>( desc.getCcuAdr() );
    addr.ccuChan_  = static_cast<uint16_t>( desc.getCcuChan() );
    addr.lldChan_  = static_cast<uint16_t>( SiStripFecKey::lldChan( desc.getI2cAddr() ) );
    addr.fedId_    = static_cast<uint16_t>( desc.getFedId() ); //@@ offset required? crate/slot needed?
    addr.feUnit_   = static_cast<uint16_t>( desc.getFeUnit() );
    addr.feChan_   = static_cast<uint16_t>( desc.getFeChan() );
  } catch (...) { handleException( __func__ ); }
  
  return addr;
}

// -----------------------------------------------------------------------------
//
std::string SiStripConfigDb::analysisType( const AnalysisType& analysis_type ) const {
  if      ( analysis_type == AnalysisDescription::T_ANALYSIS_FASTFEDCABLING ) { return "FAST_CABLING"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_TIMING )         { return "APV_TIMING"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_OPTOSCAN )       { return "OPTO_SCAN"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_PEDESTALS )      { return "PEDESTALS"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_APVLATENCY )     { return "APV_LATENCY"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_FINEDELAY )      { return "FINE_DELAY"; }
  else if ( analysis_type == AnalysisDescription::T_ANALYSIS_CALIBRATION )    { return "CALIBRATION"; }
  else if ( analysis_type == AnalysisDescription::T_UNKNOWN )                 { return "UNKNOWN ANALYSIS TYPE"; }
  else { return "UNDEFINED ANALYSIS TYPE"; }
}

#endif // USING_NEW_DATABASE_MODEL
