// Last commit: $Id: SiStripPartition.cc,v 1.3 2008/04/24 16:02:34 bainbrid Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripPartition.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <iostream>
#include <cmath>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripPartition::SiStripPartition() :
  partitionName_(""), 
  runNumber_(0),
  runType_(sistrip::UNDEFINED_RUN_TYPE),
  cabVersion_(0,0),
  fedVersion_(0,0),
  fecVersion_(0,0),
  calVersion_(0,0),
  dcuVersion_(0,0),
  forceVersions_(false),
  forceCurrentState_(true),
  inputModuleXml_(""),
  inputDcuInfoXml_(""),
  inputFecXml_(),
  inputFedXml_()
{
  reset();
}

// -----------------------------------------------------------------------------
// 
SiStripPartition::~SiStripPartition() {
  inputFecXml_.clear();
  inputFedXml_.clear();
}

// -----------------------------------------------------------------------------
// 
void SiStripPartition::reset() {
  partitionName_     = "";
  runNumber_         = 0;
  runType_           = sistrip::UNDEFINED_RUN_TYPE;
  forceVersions_     = false;
  forceCurrentState_ = false;

  cabVersion_ = std::make_pair(0,0);
  fedVersion_ = std::make_pair(0,0);
  fecVersion_ = std::make_pair(0,0);
  calVersion_ = std::make_pair(0,0);
  dcuVersion_ = std::make_pair(0,0);

  inputModuleXml_   = "";
  inputDcuInfoXml_  = "";
  inputFecXml_.clear(); inputFecXml_.push_back("");
  inputFedXml_.clear(); inputFedXml_.push_back("");
}

// -----------------------------------------------------------------------------
// 
void SiStripPartition::pset( const edm::ParameterSet& pset ) {

  partitionName_     = pset.getUntrackedParameter<std::string>( "PartitionName", "" );
  runNumber_         = pset.getUntrackedParameter<unsigned int>( "RunNumber", 0 );
  forceVersions_     = pset.getUntrackedParameter<bool>( "ForceVersions", false );
  forceCurrentState_ = pset.getUntrackedParameter<bool>( "ForceCurrentState", true );

  std::vector<unsigned int> tmp1(2,0);
  cabVersion_ = versions( pset.getUntrackedParameter< std::vector<unsigned int> >( "CablingVersion", tmp1 ) );
  fedVersion_ = versions( pset.getUntrackedParameter< std::vector<unsigned int> >( "FedVersion", tmp1 ) );
  fecVersion_ = versions( pset.getUntrackedParameter< std::vector<unsigned int> >( "FecVersion", tmp1 ) );
  dcuVersion_ = versions( pset.getUntrackedParameter< std::vector<unsigned int> >( "DcuDetIdVersion", tmp1 ) );
  calVersion_ = versions( pset.getUntrackedParameter< std::vector<unsigned int> >( "CalibVersion", tmp1 ) );
  
  std::vector<std::string> tmp2(1,"");
  inputModuleXml_   = pset.getUntrackedParameter<std::string>( "InputModuleXml", "" );
  inputDcuInfoXml_  = pset.getUntrackedParameter<std::string>( "InputDcuInfoXml", "" ); 
  inputFecXml_      = pset.getUntrackedParameter< std::vector<std::string> >( "InputFecXml", tmp2 ); 
  inputFedXml_      = pset.getUntrackedParameter< std::vector<std::string> >( "InputFedXml", tmp2 );

}

// -----------------------------------------------------------------------------
// 
void SiStripPartition::update( const SiStripConfigDb* const db ) {
  
  // Check
  if ( !db ) {
    edm::LogError(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb object!"
      << " Aborting update...";
    return;
  }
  
  // Check
  if ( !(db->deviceFactory(__func__)) ) {
    edm::LogError(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL pointer to DeviceFactory object!"
      << " Aborting update...";
    return;
  }

  // Update versions if using versions from "current state"
  if ( forceCurrentState_ || forceVersions_ ) { 
    
    // Find state for given partition
    tkStateVector states;
#ifdef USING_NEW_DATABASE_MODEL
    states = db->deviceFactory(__func__)->getCurrentStates(); 
#else
    states = *( db->deviceFactory(__func__)->getCurrentStates() ); 
#endif
    tkStateVector::const_iterator istate = states.begin();
    tkStateVector::const_iterator jstate = states.end();
    while ( istate != jstate ) {
      if ( *istate && partitionName_ == (*istate)->getPartitionName() ) { break; }
      istate++;
    }
    
    // Set versions if state was found
    if ( istate != states.end() ) {
	
#ifdef USING_NEW_DATABASE_MODEL
      if ( !cabVersion_.first &&
	   !cabVersion_.second ) { 
	cabVersion_.first = (*istate)->getConnectionVersionMajorId(); 
	cabVersion_.second = (*istate)->getConnectionVersionMinorId(); 
      }
#endif
    
      if ( !fecVersion_.first &&
	   !fecVersion_.second ) { 
	fecVersion_.first = (*istate)->getFecVersionMajorId(); 
	fecVersion_.second = (*istate)->getFecVersionMinorId(); 
      }
      if ( !fedVersion_.first &&
	   !fedVersion_.second ) { 
	fedVersion_.first = (*istate)->getFedVersionMajorId(); 
	fedVersion_.second = (*istate)->getFedVersionMinorId(); 
      }
	
#ifdef USING_NEW_DATABASE_MODEL
      if ( !dcuVersion_.first &&
	   !dcuVersion_.second ) { 
	dcuVersion_.first = (*istate)->getDcuInfoVersionMajorId(); 
	dcuVersion_.second = (*istate)->getDcuInfoVersionMinorId(); 
      }
#endif
	
#ifdef USING_NEW_DATABASE_MODEL
      /*
	if ( !psuMajor_.first && 
	!psuMajor_.second ) { 
	psuMajor_.first = (*istate)->getDcuPsuMapVersionMajorId();
	psuMinor_.second = (*istate)->getDcuPsuMapVersionMinorId(); 
	}
      */
#endif
	
#ifdef USING_NEW_DATABASE_MODEL
      /*
	if ( !calVersion_.first &&
	!calVersion_.second ) { 
	calVersion_.first = (*istate)->getAnalysisVersionMajorId(); 
	calVersion_.second = (*istate)->getAnalysisVersionMinorId(); 
	}
      */
#endif
	
    } else {
      std::stringstream ss;
      edm::LogError(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Unable to find \"current state\" for partition \""
	<< partitionName_ << "\"";
    }

  } else { // Use run number
      
    // Retrieve TkRun object for given run (0 means "latest run")
    TkRun* run = 0;    
    if ( !runNumber_ ) { run = db->deviceFactory(__func__)->getLastRun( partitionName_ ); }
    else { run = db->deviceFactory(__func__)->getRun( partitionName_, runNumber_ ); }
  
    // Retrieve versioning for given TkRun object 
    if ( run ) {
      
      if ( run->getRunNumber() ) {
	
	if ( !runNumber_ ) { runNumber_ = run->getRunNumber(); }
      
	if ( runNumber_ == run->getRunNumber() ) {
	
#ifdef USING_NEW_DATABASE_MODEL
	  cabVersion_.first = run->getConnectionVersionMajorId(); 
	  cabVersion_.second = run->getConnectionVersionMinorId(); 
#endif
	  
	  fecVersion_.first = run->getFecVersionMajorId(); 
	  fecVersion_.second = run->getFecVersionMinorId(); 

	  fedVersion_.first = run->getFedVersionMajorId(); 
	  fedVersion_.second = run->getFedVersionMinorId(); 
	  
#ifdef USING_NEW_DATABASE_MODEL
	  dcuVersion_.first = run->getDcuInfoVersionMajorId(); 
	  dcuVersion_.second = run->getDcuInfoVersionMinorId(); 
#endif

#ifdef USING_NEW_DATABASE_MODEL
	  //@@ psuMajor_.first = run->getDcuPsuMapVersionMajorId(); 
	  //@@ psuMinor_.second = run->getDcuPsuMapVersionMinorId(); 
#endif

#ifdef USING_NEW_DATABASE_MODEL
	  //@@ calVersion_.first = run->getAnalysisVersionMajorId(); 
	  //@@ calVersion_.second = run->getAnalysisVersionMinorId(); 
#endif
	
	} else {
	  edm::LogError(mlConfigDb_)
	    << "[SiStripConfigDb::" << __func__ << "]"
	    << " Mismatch of run number requested (" 
	    << runNumber_
	    << ") and received (" 
	    << run->getRunNumber() << ")"
	    << " to/from database for partition \"" 
	    << partitionName_ << "\"";
	}

      } else {
	edm::LogError(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL run number returned!"
	  << " for partition \"" << partitionName_ << "\"";
      }

      uint16_t type = run->getModeId( run->getMode() );
      if      ( type ==  1 ) { runType_ = sistrip::PHYSICS; }
      else if ( type ==  2 ) { runType_ = sistrip::PEDESTALS; }
      else if ( type ==  3 ) { runType_ = sistrip::CALIBRATION; }
      else if ( type == 33 ) { runType_ = sistrip::CALIBRATION_DECO; }
      else if ( type ==  4 ) { runType_ = sistrip::OPTO_SCAN; }
      else if ( type ==  5 ) { runType_ = sistrip::APV_TIMING; }
      else if ( type ==  6 ) { runType_ = sistrip::APV_LATENCY; }
      else if ( type ==  7 ) { runType_ = sistrip::FINE_DELAY_PLL; }
      else if ( type ==  8 ) { runType_ = sistrip::FINE_DELAY_TTC; }
      else if ( type == 10 ) { runType_ = sistrip::MULTI_MODE; }
      else if ( type == 12 ) { runType_ = sistrip::FED_TIMING; }
      else if ( type == 13 ) { runType_ = sistrip::FED_CABLING; }
      else if ( type == 14 ) { runType_ = sistrip::VPSP_SCAN; }
      else if ( type == 15 ) { runType_ = sistrip::DAQ_SCOPE_MODE; }
      else if ( type == 16 ) { runType_ = sistrip::QUITE_FAST_CABLING; }
      else if ( type == 21 ) { runType_ = sistrip::FAST_CABLING; }
      else if ( type ==  0 ) { 
	runType_ = sistrip::UNDEFINED_RUN_TYPE;
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL run type returned!"
	  << " for partition \"" << partitionName_ << "\"";
      } else { 
	runType_ = sistrip::UNKNOWN_RUN_TYPE; 
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " UNKONWN run type (" << type<< ") returned!"
	  << " for partition \"" << partitionName_ << "\"";
      }
      
    } else {
      edm::LogError(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " NULL pointer to TkRun object!"
	<< " Unable to retrieve versions for run number "
	<< runNumber_
	<< ". Run number may not be consistent with partition \"" 
	<< partitionName_ << "\"!"; //@@ only using first here!!!
    }

  }

}

// -----------------------------------------------------------------------------
// 
void SiStripPartition::print( std::stringstream& ss, bool using_db ) const {

  ss << "  Partition                 : " << partitionName_ << std::endl;
  
  if ( using_db ) {
    
    ss << "  Run number                : ";
    if ( forceCurrentState_ )  { ss << "Forced \"current state\"! (equivalent to versions below)"; }
    else if ( forceVersions_ ) { ss << "Forced versions specified below!"; }
    else /* use run number */  { ss << runNumber_; }
    
    ss << std::endl;
    if ( !forceVersions_ ) { 
      ss << "  Run type                  : " << SiStripEnumsAndStrings::runType( runType_ ) << std::endl;
    }
    
    ss << "  Cabling major/minor vers  : " << cabVersion_.first << "." << cabVersion_.second << std::endl
       << "  FEC major/minor vers      : " << fecVersion_.first << "." << fecVersion_.second << std::endl
       << "  FED major/minor vers      : " << fedVersion_.first << "." << fedVersion_.second << std::endl
       << "  DCU-DetId maj/min vers    : " << dcuVersion_.first << "." << dcuVersion_.second << std::endl
       << "  Calibration maj/min vers  : " << calVersion_.first << "." << calVersion_.second << std::endl;
    
  } else {
    
    ss << "  Input \"module.xml\" file   : " << inputModuleXml_ << std::endl
       << "  Input \"dcuinfo.xml\" file  : " << inputDcuInfoXml_ << std::endl
       << "  Input \"fec.xml\" file(s)   : ";
    std::vector<std::string>::const_iterator ifec = inputFecXml_.begin();
    for ( ; ifec != inputFecXml_.end(); ifec++ ) { ss << *ifec << ", "; }
    ss << std::endl;
    ss << "  Input \"fed.xml\" file(s)   : ";
    std::vector<std::string>::const_iterator ifed = inputFedXml_.begin();
    for ( ; ifed != inputFedXml_.end(); ifed++ ) { ss << *ifed << ", "; }
    ss << std::endl;
    
  }
  
}

// -----------------------------------------------------------------------------
// 
std::ostream& operator<< ( std::ostream& os, const SiStripPartition& params ) {
  std::stringstream ss;
  params.print(ss);
  os << ss.str();
  return os;
}

// -----------------------------------------------------------------------------
// 
SiStripPartition::Versions SiStripPartition::versions( std::vector<unsigned int> input ) {
  if ( input.size() != 2 ) { 
    edm::LogWarning(mlConfigDb_)
      << "[SiStripPartition::" << __func__ << "]"
      << " Unexpected size (" << input.size()
      << ") for  vector containing version numbers (major,minor)!"
      << " Resizing to 2 elements (default values will be 0,0)...";
    input.resize(2,0);
  }
  return std::make_pair( input[0], input[1] );
}
