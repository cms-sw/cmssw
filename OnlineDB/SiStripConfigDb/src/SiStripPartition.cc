// Last commit: $Id: SiStripPartition.cc,v 1.20 2010/04/20 09:28:35 dstrom Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripPartition.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <iostream>
#include <cmath>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
std::string SiStripPartition::defaultPartitionName_ = "DefaultPartition";

// -----------------------------------------------------------------------------
// 
SiStripPartition::SiStripPartition() :
  partitionName_(defaultPartitionName_), 
  runNumber_(0),
  runType_(sistrip::UNDEFINED_RUN_TYPE),
  forceVersions_(false),
  forceCurrentState_(false),
  cabVersion_(0,0),
  fedVersion_(0,0),
  fecVersion_(0,0),
  dcuVersion_(0,0),
  psuVersion_(0,0),
  //#ifdef USING_DATABASE_MASKING
  maskVersion_(0,0),
				  //#endif
  globalAnalysisV_(0),
  runTableVersion_(0,0),
  fastCablingV_(0,0),
  apvTimingV_(0,0),
  optoScanV_(0,0),
  vpspScanV_(0,0),
  apvCalibV_(0,0),
  pedestalsV_(0,0),
  apvLatencyV_(0,0),
  fineDelayV_(0,0),
  inputModuleXml_(""),
  inputDcuInfoXml_(""),
  inputFecXml_(),
  inputFedXml_()
{;}

// -----------------------------------------------------------------------------
// 
SiStripPartition::SiStripPartition( std::string partition ) :
  partitionName_(partition), 
  runNumber_(0),
  runType_(sistrip::UNDEFINED_RUN_TYPE),
  forceVersions_(false),
  forceCurrentState_(false),
  cabVersion_(0,0),
  fedVersion_(0,0),
  fecVersion_(0,0),
  dcuVersion_(0,0),
  psuVersion_(0,0),
  //#ifdef USING_DATABASE_MASKING
  maskVersion_(0,0),
				  //#endif
  globalAnalysisV_(0),
  runTableVersion_(0,0),
  fastCablingV_(0,0),
  apvTimingV_(0,0),
  optoScanV_(0,0),
  vpspScanV_(0,0),
  apvCalibV_(0,0),
  pedestalsV_(0,0),
  apvLatencyV_(0,0),
  fineDelayV_(0,0),
  inputModuleXml_(""),
  inputDcuInfoXml_(""),
  inputFecXml_(),
  inputFedXml_()
{
  if ( partitionName_.empty() ) { partitionName_ = defaultPartitionName_; }
}

// -----------------------------------------------------------------------------
// 
SiStripPartition::SiStripPartition( const SiStripPartition& input ) :
  partitionName_( input.partitionName() ), 
  runNumber_( input.runNumber() ),
  runType_( input.runType() ),
  forceVersions_( input.forceVersions() ),
  forceCurrentState_( input.forceCurrentState() ),
  cabVersion_( input.cabVersion() ),
  fedVersion_( input.fedVersion() ),
  fecVersion_( input.fecVersion() ),
  dcuVersion_( input.dcuVersion() ),
  psuVersion_( input.psuVersion() ),
  //#ifdef USING_DATABASE_MASKING
  maskVersion_( input.maskVersion() ),
				  //#endif
  globalAnalysisV_( input.globalAnalysisVersion() ),
  runTableVersion_( input.runTableVersion() ),
  fastCablingV_( input.fastCablingVersion() ),
  apvTimingV_( input.apvTimingVersion() ),
  optoScanV_( input.optoScanVersion() ),
  vpspScanV_( input.vpspScanVersion() ),
  apvCalibV_( input.apvCalibVersion() ),
  pedestalsV_( input.pedestalsVersion() ),
  apvLatencyV_( input.apvLatencyVersion() ),
  fineDelayV_( input.fineDelayVersion() ),
  inputModuleXml_( input.inputModuleXml() ),
  inputDcuInfoXml_( input.inputDcuInfoXml() ),
  inputFecXml_( input.inputFecXml() ),
  inputFedXml_( input.inputFedXml() )
{;}

// -----------------------------------------------------------------------------
// 
SiStripPartition& SiStripPartition::operator= ( const SiStripPartition& input ){
  if ( this == &input ) { return *this; } // self-assignment
  partitionName_ = input.partitionName(); 
  runNumber_ = input.runNumber();
  runType_ = input.runType();
  forceVersions_ = input.forceVersions();
  forceCurrentState_ = input.forceCurrentState();
  cabVersion_ = input.cabVersion();
  fedVersion_ = input.fedVersion();
  fecVersion_ = input.fecVersion();
  dcuVersion_ = input.dcuVersion();
  psuVersion_ = input.psuVersion();
  //#ifdef USING_DATABASE_MASKING
  maskVersion_ = input.maskVersion();
  //#endif
  globalAnalysisV_ = input.globalAnalysisVersion();
  runTableVersion_ = input.runTableVersion();
  fastCablingV_ = input.fastCablingVersion();
  apvTimingV_ = input.apvTimingVersion();
  optoScanV_ = input.optoScanVersion();
  vpspScanV_ = input.vpspScanVersion();
  apvCalibV_ = input.apvCalibVersion();
  pedestalsV_ = input.pedestalsVersion();
  apvLatencyV_ = input.apvLatencyVersion();
  fineDelayV_ = input.fineDelayVersion();
  inputModuleXml_ = input.inputModuleXml();
  inputDcuInfoXml_ = input.inputDcuInfoXml();
  inputFecXml_ = input.inputFecXml();
  inputFedXml_ = input.inputFedXml();
  return *this;
}

// -----------------------------------------------------------------------------
// 
bool SiStripPartition::operator== ( const SiStripPartition& input ) const {
  return ( partitionName_ == input.partitionName() && 
	   runNumber_ == input.runNumber() &&
	   runType_ == input.runType() &&
	   forceVersions_ == input.forceVersions() &&
	   forceCurrentState_ == input.forceCurrentState() &&
	   cabVersion_ == input.cabVersion() &&
	   fedVersion_ == input.fedVersion() &&
	   fecVersion_ == input.fecVersion() &&
	   dcuVersion_ == input.dcuVersion() &&
	   psuVersion_ == input.psuVersion() &&
	   //#ifdef USING_DATABASE_MASKING
	   maskVersion_ == input.maskVersion() &&
	   //#endif
	   globalAnalysisV_ == input.globalAnalysisVersion() &&
	   runTableVersion_ == input.runTableVersion() &&
	   fastCablingV_ == input.fastCablingVersion() &&
	   apvTimingV_ == input.apvTimingVersion() &&
	   optoScanV_ == input.optoScanVersion() &&
	   vpspScanV_ == input.vpspScanVersion() &&
	   apvCalibV_ == input.apvCalibVersion() &&
	   pedestalsV_ == input.pedestalsVersion() &&
	   apvLatencyV_ == input.apvLatencyVersion() &&
	   fineDelayV_ == input.fineDelayVersion() &&
	   inputModuleXml_ == input.inputModuleXml() &&
	   inputDcuInfoXml_ == input.inputDcuInfoXml() &&
	   inputFecXml_ == input.inputFecXml() &&
	   inputFedXml_ == input.inputFedXml() );
}

// -----------------------------------------------------------------------------
// 
bool SiStripPartition::operator!= ( const SiStripPartition& input ) const {
  return !( *this == input );
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
  dcuVersion_ = std::make_pair(0,0);
  psuVersion_ = std::make_pair(0,0);
  //#ifdef USING_DATABASE_MASKING
  maskVersion_ = std::make_pair(0,0);
  //#endif
  
  globalAnalysisV_ = 0;
  runTableVersion_ = std::make_pair(0,0);
  fastCablingV_ = std::make_pair(0,0);
  apvTimingV_   = std::make_pair(0,0);
  optoScanV_    = std::make_pair(0,0);
  vpspScanV_    = std::make_pair(0,0);
  apvCalibV_    = std::make_pair(0,0);
  pedestalsV_   = std::make_pair(0,0);
  apvLatencyV_  = std::make_pair(0,0);
  fineDelayV_   = std::make_pair(0,0);
  
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
  forceCurrentState_ = pset.getUntrackedParameter<bool>( "ForceCurrentState", false );

  std::vector<uint32_t> tmp1(2,0);
  cabVersion_ = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "CablingVersion", tmp1 ) );
  fedVersion_ = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "FedVersion", tmp1 ) );
  fecVersion_ = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "FecVersion", tmp1 ) );
  dcuVersion_ = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "DcuDetIdsVersion", tmp1 ) );
  psuVersion_ = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "DcuPsuMapVersion", tmp1 ) );
  //#ifdef USING_DATABASE_MASKING
  maskVersion_ = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "MaskVersion", tmp1 ) );
  //#endif

  std::vector<uint32_t> tmp2(2,0);
  globalAnalysisV_ = pset.getUntrackedParameter<uint32_t>( "GlobalAnalysisVersion", 0 );
  fastCablingV_    = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "FastCablingVersion", tmp2 ) );
  apvTimingV_      = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "ApvTimingVersion", tmp2 ) );
  optoScanV_       = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "OptoScanVersion", tmp2 ) );
  vpspScanV_       = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "VpspScanVersion", tmp2 ) );
  apvCalibV_       = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "ApvCalibVersion", tmp2 ) );
  pedestalsV_      = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "PedestalsVersion", tmp2 ) );
  apvLatencyV_     = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "ApvLatencyVersion", tmp2 ) );
  fineDelayV_      = versions( pset.getUntrackedParameter< std::vector<uint32_t> >( "FineDelayVersion", tmp2 ) );
  
  std::vector<std::string> tmp3(1,"");
  inputModuleXml_   = pset.getUntrackedParameter<std::string>( "InputModuleXml", "" );
  inputDcuInfoXml_  = pset.getUntrackedParameter<std::string>( "InputDcuInfoXml", "" ); 
  inputFecXml_      = pset.getUntrackedParameter< std::vector<std::string> >( "InputFecXml", tmp3 ); 
  inputFedXml_      = pset.getUntrackedParameter< std::vector<std::string> >( "InputFedXml", tmp3 );
  
}

// -----------------------------------------------------------------------------
// 
void SiStripPartition::update( const SiStripConfigDb* const db ) {
  
  // Check
  if ( !db ) {
    edm::LogError(mlConfigDb_)
      << "[SiStripPartition::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb object!"
      << " Aborting update...";
    return;
  }
  
  // Check
  DeviceFactory* const df = db->deviceFactory(__func__);
  if ( !df ) {
    edm::LogError(mlConfigDb_)
      << "[SiStripPartition::" << __func__ << "]"
      << " NULL pointer to DeviceFactory object!"
      << " Aborting update...";
    return;
  }

  // Check partition name
  if ( partitionName_ == defaultPartitionName_ ) { return; }

  // Debug
  std::stringstream ss;
  ss << "[SiStripPartition::" << __func__ << "]"
     << " Updating description versions for partition \""
     << partitionName_
     << "\"...";
  LogTrace(mlConfigDb_) << ss.str();

  try {

    // Reset container indicating state versions for commissioning runs 
    runTableVersion_ = Versions(0,0);

    // Update versions if using versions from "current state"
    if ( forceCurrentState_ || forceVersions_ ) { 
    
      // Find state for given partition
      tkStateVector states;
      states = df->getCurrentStates(); 
      tkStateVector::const_iterator istate = states.begin();
      tkStateVector::const_iterator jstate = states.end();
      while ( istate != jstate ) {
	if ( *istate && partitionName_ == (*istate)->getPartitionName() ) { break; }
	istate++;
      }
      
      // Set versions if state was found
      if ( istate != states.end() ) {
	
	if ( !cabVersion_.first &&
	     !cabVersion_.second ) { 
	  cabVersion_.first = (*istate)->getConnectionVersionMajorId(); 
	  cabVersion_.second = (*istate)->getConnectionVersionMinorId(); 
	}
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
	if ( !dcuVersion_.first &&
	     !dcuVersion_.second ) { 
	  dcuVersion_.first = (*istate)->getDcuInfoVersionMajorId(); 
	  dcuVersion_.second = (*istate)->getDcuInfoVersionMinorId(); 
	}
	if ( !psuVersion_.first &&
	     !psuVersion_.second ) { 
	  psuVersion_.first = (*istate)->getDcuPsuMapVersionMajorId();
	  psuVersion_.second = (*istate)->getDcuPsuMapVersionMinorId(); 
	}
	//#ifdef USING_DATABASE_MASKING
	if ( !forceVersions_ &&
	     !maskVersion_.first &&
	     !maskVersion_.second ) { 
	  maskVersion_.first = (*istate)->getMaskVersionMajorId();
	  maskVersion_.second = (*istate)->getMaskVersionMinorId(); 
	}
	//#endif
	
	// Retrieve global and local versions 
	if ( forceCurrentState_ || globalAnalysisV_ ) { // use global version (or current state)

	  // Set global version
	  if ( forceCurrentState_ ) { globalAnalysisV_ = (*istate)->getAnalysisVersionMapPointerId(); }
	  
	  // Retrieve local versions
	  HashMapAnalysisVersions local_versions = df->getLocalAnalysisVersions( globalAnalysisV_ );
	  
	  // Iterate through map< AnalysisType, pair<Major,Minor> >
	  HashMapAnalysisVersions::const_iterator ivers = local_versions.begin();
	  HashMapAnalysisVersions::const_iterator jvers = local_versions.end();
	  for ( ; ivers != jvers; ++ivers ) {
	    if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_FASTFEDCABLING ) { 
	      fastCablingV_.first = ivers->second.first;
	      fastCablingV_.second = ivers->second.second;
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_TIMING ) {
	      apvTimingV_.first = ivers->second.first;
	      apvTimingV_.second = ivers->second.second;
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_OPTOSCAN ) {
	      optoScanV_.first = ivers->second.first;
	      optoScanV_.second = ivers->second.second;
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_VPSPSCAN ) {
	      vpspScanV_.first = ivers->second.first;
	      vpspScanV_.second = ivers->second.second;
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_CALIBRATION ) {
	      apvCalibV_.first = ivers->second.first;
	      apvCalibV_.second = ivers->second.second;
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_PEDESTALS ) {
	      pedestalsV_.first = ivers->second.first;
	      pedestalsV_.second = ivers->second.second;
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_APVLATENCY ) {
	      apvLatencyV_.first = ivers->second.first;
	      apvLatencyV_.second = ivers->second.second;
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_FINEDELAY ) {
	      fineDelayV_.first = ivers->second.first;
	      fineDelayV_.second = ivers->second.second;
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_UNKNOWN ) {
	      std::stringstream ss;
	      edm::LogError(mlConfigDb_)
		<< "[SiStripPartition::" << __func__ << "]"
		<< " Found UNKNOWN AnalysisType!";
	      edm::LogWarning(mlConfigDb_) << ss.str();
	    } else {
	      std::stringstream ss;
	      edm::LogError(mlConfigDb_)
		<< "[SiStripPartition::" << __func__ << "]"
		<< " Unable to match content to any AnalysisType!";
	      edm::LogWarning(mlConfigDb_) << ss.str();
	    }
	  }
      
	} else if ( !globalAnalysisV_ ) { // use local versions
	
	  // Retrieve local versions and set if necessary
	  globalAnalysisV_ = (*istate)->getAnalysisVersionMapPointerId(); 
	  HashMapAnalysisVersions local_versions = df->getLocalAnalysisVersions( globalAnalysisV_ );
	  
	  // Iterate through map< AnalysisType, pair<Major,Minor> >
	  HashMapAnalysisVersions::const_iterator ivers = local_versions.begin();
	  HashMapAnalysisVersions::const_iterator jvers = local_versions.end();
	  for ( ; ivers != jvers; ++ivers ) {
	    if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_FASTFEDCABLING ) { 
	      if ( !fastCablingV_.first && !fastCablingV_.second ) {
		fastCablingV_.first = ivers->second.first;
		fastCablingV_.second = ivers->second.second;
	      }
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_TIMING ) {
	      if ( !apvTimingV_.first && !apvTimingV_.second ) {
		apvTimingV_.first = ivers->second.first;
		apvTimingV_.second = ivers->second.second;
	      }
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_OPTOSCAN ) {
	      if ( !optoScanV_.first && !optoScanV_.second ) {
		optoScanV_.first = ivers->second.first;
		optoScanV_.second = ivers->second.second;
	      }
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_VPSPSCAN ) {
	      if ( !vpspScanV_.first && !vpspScanV_.second ) {
		vpspScanV_.first = ivers->second.first;
		vpspScanV_.second = ivers->second.second;
	      }
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_CALIBRATION ) {
	      if ( !apvCalibV_.first && !apvCalibV_.second ) {
		apvCalibV_.first = ivers->second.first;
		apvCalibV_.second = ivers->second.second;
	      }
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_PEDESTALS ) {
	      if ( !pedestalsV_.first && !pedestalsV_.second ) {
		pedestalsV_.first = ivers->second.first;
		pedestalsV_.second = ivers->second.second;
	      }
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_APVLATENCY ) {
	      if ( !apvLatencyV_.first && !apvLatencyV_.second ) {
		apvLatencyV_.first = ivers->second.first;
		apvLatencyV_.second = ivers->second.second;
	      }
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_FINEDELAY ) {
	      if ( !fineDelayV_.first && !fineDelayV_.second ) {
		fineDelayV_.first = ivers->second.first;
		fineDelayV_.second = ivers->second.second;
	      }
	    } else if ( ivers->first == CommissioningAnalysisDescription::T_UNKNOWN ) {
	      std::stringstream ss;
	      edm::LogError(mlConfigDb_)
		<< "[SiStripPartition::" << __func__ << "]"
		<< " Found UNKNOWN AnalysisType!";
	      edm::LogWarning(mlConfigDb_) << ss.str();
	    } else {
	      std::stringstream ss;
	      edm::LogError(mlConfigDb_)
		<< "[SiStripPartition::" << __func__ << "]"
		<< " Unable to match content to any AnalysisType!";
	      edm::LogWarning(mlConfigDb_) << ss.str();
	    }
	  }
      
	}

      } else {
	std::stringstream ss;
	edm::LogError(mlConfigDb_)
	  << "[SiStripPartition::" << __func__ << "]"
	  << " Unable to find \"current state\" for partition \""
	  << partitionName_ << "\"";
      }
    
    } else { // ---------- USE RUN NUMBER ----------
    
      // Retrieve TkRun object for given run (0 means "latest run")
      TkRun* run = 0;
      if ( !runNumber_ ) { run = df->getLastRun( partitionName_ ); }
      else { run = df->getRun( partitionName_, runNumber_ ); }
  
      // Retrieve versioning for given TkRun object 
      if ( run ) {
	
	if ( run->getRunNumber() ) {
	  
	  if ( !runNumber_ ) { runNumber_ = run->getRunNumber(); }
	  
	  if ( runNumber_ == run->getRunNumber() ) {
	    
	    cabVersion_.first = run->getConnectionVersionMajorId(); 
	    cabVersion_.second = run->getConnectionVersionMinorId(); 
	    
	    fecVersion_.first = run->getFecVersionMajorId(); 
	    fecVersion_.second = run->getFecVersionMinorId(); 
	    
	    fedVersion_.first = run->getFedVersionMajorId(); 
	    fedVersion_.second = run->getFedVersionMinorId(); 
	    
	    dcuVersion_.first = run->getDcuInfoVersionMajorId(); 
	    dcuVersion_.second = run->getDcuInfoVersionMinorId(); 
	    
	    psuVersion_.first = run->getDcuPsuMapVersionMajorId();
	    psuVersion_.second = run->getDcuPsuMapVersionMinorId(); 
	    
	    //#ifdef USING_DATABASE_MASKING
	    maskVersion_.first = run->getMaskVersionMajorId();
	    maskVersion_.second = run->getMaskVersionMinorId(); 
	    //#endif
	    
	    // Check run type
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
		<< "[SiStripPartition::" << __func__ << "]"
		<< " NULL run type returned!"
		<< " for partition \"" << partitionName_ << "\"";
	    } else { 
	      runType_ = sistrip::UNKNOWN_RUN_TYPE; 
	      edm::LogWarning(mlConfigDb_)
		<< "[SiStripPartition::" << __func__ << "]"
		<< " UNKNOWN run type (" << type<< ") returned!"
		<< " for partition \"" << partitionName_ << "\"";
	    }

	    // Retrieve global and local versions from state associated with given run
	    globalAnalysisV_ = run->getAnalysisVersionMapPointerId(); 
	    HashMapAnalysisVersions local_versions = df->getLocalAnalysisVersions( globalAnalysisV_ );
	    HashMapAnalysisVersions::const_iterator ivers = local_versions.begin();
	    HashMapAnalysisVersions::const_iterator jvers = local_versions.end();
	    for ( ; ivers != jvers; ++ivers ) {
	      if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_FASTFEDCABLING ) { 
		fastCablingV_.first = ivers->second.first;
		fastCablingV_.second = ivers->second.second;
	      } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_TIMING ) {
		apvTimingV_.first = ivers->second.first;
		apvTimingV_.second = ivers->second.second;
	      } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_OPTOSCAN ) {
		optoScanV_.first = ivers->second.first;
		optoScanV_.second = ivers->second.second;
	      } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_VPSPSCAN ) {
		vpspScanV_.first = ivers->second.first;
		vpspScanV_.second = ivers->second.second;
	      } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_CALIBRATION ) {
		apvCalibV_.first = ivers->second.first;
		apvCalibV_.second = ivers->second.second;
	      } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_PEDESTALS ) {
		pedestalsV_.first = ivers->second.first;
		pedestalsV_.second = ivers->second.second;
	      } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_APVLATENCY ) {
		apvLatencyV_.first = ivers->second.first;
		apvLatencyV_.second = ivers->second.second;
	      } else if ( ivers->first == CommissioningAnalysisDescription::T_ANALYSIS_FINEDELAY ) {
		fineDelayV_.first = ivers->second.first;
		fineDelayV_.second = ivers->second.second;
	      } else if ( ivers->first == CommissioningAnalysisDescription::T_UNKNOWN ) {
		std::stringstream ss;
		edm::LogError(mlConfigDb_)
		  << "[SiStripPartition::" << __func__ << "]"
		  << " Found UNKNOWN AnalysisType!";
		edm::LogWarning(mlConfigDb_) << ss.str();
	      } else {
		std::stringstream ss;
		edm::LogError(mlConfigDb_)
		  << "[SiStripPartition::" << __func__ << "]"
		  << " Unable to match content to any AnalysisType!";
		edm::LogWarning(mlConfigDb_) << ss.str();
	      }
	    }

	    // If commissioning run, override version of analysis type with version PRODUCED by run (from history)
	    if ( runType_ != sistrip::PHYSICS &&
		 runType_ != sistrip::UNDEFINED_RUN_TYPE &&
		 runType_ != sistrip::UNKNOWN_RUN_TYPE ) { 
	      
	      // Determine analysis type from run type
	      CommissioningAnalysisDescription::commissioningType type = CommissioningAnalysisDescription::T_UNKNOWN;
	      if      ( runType_ == sistrip::FAST_CABLING )   { type = CommissioningAnalysisDescription::T_ANALYSIS_FASTFEDCABLING; }
	      else if ( runType_ == sistrip::APV_TIMING )     { type = CommissioningAnalysisDescription::T_ANALYSIS_TIMING; }
	      else if ( runType_ == sistrip::OPTO_SCAN )      { type = CommissioningAnalysisDescription::T_ANALYSIS_OPTOSCAN; }
	      else if ( runType_ == sistrip::VPSP_SCAN )      { type = CommissioningAnalysisDescription::T_ANALYSIS_VPSPSCAN; }
	      else if ( runType_ == sistrip::CALIBRATION )    { type = CommissioningAnalysisDescription::T_ANALYSIS_CALIBRATION; }
	      else if ( runType_ == sistrip::PEDESTALS )      { type = CommissioningAnalysisDescription::T_ANALYSIS_PEDESTALS; }
	      else if ( runType_ == sistrip::APV_LATENCY )    { type = CommissioningAnalysisDescription::T_ANALYSIS_APVLATENCY; }
	      else if ( runType_ == sistrip::FINE_DELAY_TTC ) { type = CommissioningAnalysisDescription::T_ANALYSIS_FINEDELAY; }

	      // Retrieve local versions for COMMISSIONING runs from history
	      HashMapRunVersion local_versions = df->getAnalysisHistory( partitionName_, type );

	      // Iterate through map< RunNumber, vector< pair<Major,Minor> > > to find appropriate run
	      HashMapRunVersion::const_iterator ivers = local_versions.end();
	      if ( runNumber_ == 0 ) { ivers = --(local_versions.end()); }
	      else { ivers = local_versions.find( runNumber_ ); } 
	      
	      // Set appropriate versions
	      if ( ivers != local_versions.end() ) {
		if ( type == CommissioningAnalysisDescription::T_ANALYSIS_FASTFEDCABLING ) { 
		  runTableVersion_ = fastCablingV_;
		  fastCablingV_.first = ivers->second.back().first;
		  fastCablingV_.second = ivers->second.back().second;

		  //@@

		} else if ( type == CommissioningAnalysisDescription::T_ANALYSIS_TIMING ) {
		  runTableVersion_ = apvTimingV_;
		  apvTimingV_.first = ivers->second.back().first;
		  apvTimingV_.second = ivers->second.back().second;
		} else if ( type == CommissioningAnalysisDescription::T_ANALYSIS_OPTOSCAN ) {
		  runTableVersion_ = optoScanV_;
		  optoScanV_.first = ivers->second.back().first;
		  optoScanV_.second = ivers->second.back().second;
		} else if ( type == CommissioningAnalysisDescription::T_ANALYSIS_VPSPSCAN ) {
		  runTableVersion_ = vpspScanV_;
		  vpspScanV_.first = ivers->second.back().first;
		  vpspScanV_.second = ivers->second.back().second;
		} else if ( type == CommissioningAnalysisDescription::T_ANALYSIS_CALIBRATION ) {
		  runTableVersion_ = apvCalibV_;
		  apvCalibV_.first = ivers->second.back().first;
		  apvCalibV_.second = ivers->second.back().second;
		} else if ( type == CommissioningAnalysisDescription::T_ANALYSIS_PEDESTALS ) {
		  runTableVersion_ = pedestalsV_;
		  pedestalsV_.first = ivers->second.back().first;
		  pedestalsV_.second = ivers->second.back().second;
		} else if ( type == CommissioningAnalysisDescription::T_ANALYSIS_APVLATENCY ) {
		  runTableVersion_ = apvLatencyV_;
		  apvLatencyV_.first = ivers->second.back().first;
		  apvLatencyV_.second = ivers->second.back().second;
		} else if ( type == CommissioningAnalysisDescription::T_ANALYSIS_FINEDELAY ) {
		  runTableVersion_ = fineDelayV_;
		  fineDelayV_.first = ivers->second.back().first;
		  fineDelayV_.second = ivers->second.back().second;
		} else if ( type == CommissioningAnalysisDescription::T_UNKNOWN ) {
		  std::stringstream ss;
		  edm::LogError(mlConfigDb_)
		    << "[SiStripPartition::" << __func__ << "]"
		    << " Found UNKNOWN AnalysisType!";
		  edm::LogWarning(mlConfigDb_) << ss.str();
		} else {
		  std::stringstream ss;
		  edm::LogError(mlConfigDb_)
		    << "[SiStripPartition::" << __func__ << "]"
		    << " Unable to match content to any AnalysisType!";
		  edm::LogWarning(mlConfigDb_) << ss.str();
		}
	      } else {
		std::stringstream ss;
		edm::LogError(mlConfigDb_)
		  << "[SiStripPartition::" << __func__ << "]"
		  << " Unable to find run number " << runNumber_
		  << " in \"history\" hash map ";
		edm::LogWarning(mlConfigDb_) << ss.str();
	      }

	    }
	
	  } else {
	    edm::LogError(mlConfigDb_)
	      << "[SiStripPartition::" << __func__ << "]"
	      << " Mismatch of run number requested (" 
	      << runNumber_
	      << ") and received (" 
	      << run->getRunNumber() << ")"
	      << " to/from database for partition \"" 
	      << partitionName_ << "\"";
	  }

	} else {
	  edm::LogError(mlConfigDb_)
	    << "[SiStripPartition::" << __func__ << "]"
	    << " NULL run number returned!"
	    << " for partition \"" << partitionName_ << "\"";
	}
      
      } else {
	edm::LogError(mlConfigDb_)
	  << "[SiStripPartition::" << __func__ << "]"
	  << " NULL pointer to TkRun object!"
	  << " Unable to retrieve versions for run number "
	  << runNumber_
	  << ". Run number may not be consistent with partition \"" 
	  << partitionName_ << "\"!"; //@@ only using first here!!!
      }

    }

  } catch (...) { db->handleException( __func__, "" ); }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripPartition::print( std::stringstream& ss, bool using_db ) const {

  ss << "  Partition                  : \"" << partitionName_ << "\"" << std::endl;
  
  if ( using_db ) {
    
    ss << "  Run number                 : ";
    if ( forceCurrentState_ )  { ss << "Forced \"current state\"! (equivalent to versions below)"; }
    else if ( forceVersions_ ) { ss << "Forced versions specified below!"; }
    else /* use run number */  { ss << runNumber_; }
    
    ss << std::endl;
    if ( !forceVersions_ ) { 
      ss << "  Run type                   : " << SiStripEnumsAndStrings::runType( runType_ ) << std::endl;
    }
    
    ss << "  Cabling major/minor vers   : " << cabVersion_.first << "." << cabVersion_.second << std::endl
       << "  FEC major/minor vers       : " << fecVersion_.first << "." << fecVersion_.second << std::endl
       << "  FED major/minor vers       : " << fedVersion_.first << "." << fedVersion_.second << std::endl
       << "  DCU-DetId map maj/min vers : " << dcuVersion_.first << "." << dcuVersion_.second << std::endl
      //#ifdef USING_DATABASE_MASKING
       << "  DCU-PSU map maj/min vers   : " << psuVersion_.first << "." << psuVersion_.second << std::endl
       << "  Mask maj/min vers          : " << maskVersion_.first << "." << maskVersion_.second << std::endl;
    //#else
    //       << "  DCU-PSU map maj/min vers   : " << psuVersion_.first << "." << psuVersion_.second << std::endl;
    //#endif

    ss << "  Global analysis version    : " << globalAnalysisV_ << std::endl;


    if ( runType_ == sistrip::PHYSICS ||
	 runType_ == sistrip::UNDEFINED_RUN_TYPE ||
	 runType_ == sistrip::UNKNOWN_RUN_TYPE ) { 
      
      ss << "  FED cabling maj/min vers   : " << fastCablingV_.first << "." << fastCablingV_.second << std::endl;
      ss << "  APV timing maj/min vers    : " << apvTimingV_.first << "." << apvTimingV_.second << std::endl;
      ss << "  Opto scan maj/min vers     : " << optoScanV_.first << "." << optoScanV_.second << std::endl;
      ss << "  VPSP scan maj/min vers     : " << vpspScanV_.first << "." << vpspScanV_.second << std::endl;
      ss << "  APV calib maj/min vers     : " << apvCalibV_.first << "." << apvCalibV_.second << std::endl;
      ss << "  Pedestals maj/min vers     : " << pedestalsV_.first << "." << pedestalsV_.second << std::endl;
      ss << "  APV latency maj/min vers   : " << apvLatencyV_.first << "." << apvLatencyV_.second << std::endl;
      ss << "  Fine delay maj/min vers    : " << fineDelayV_.first << "." << fineDelayV_.second << std::endl;
      
    } else {
      
      if ( runType_ != sistrip::FAST_CABLING ) { 
	ss << "  FED cabling maj/min vers   : " << fastCablingV_.first << "." << fastCablingV_.second << std::endl;
      } else {
	ss << "  FED cabling maj/min vers   : " << runTableVersion_.first << "." << runTableVersion_.second
	   << " <= This \"state\" version overriden by \"history\" version " 
	   << fastCablingV_.first << "." << fastCablingV_.second 
	   << " for this FED cabling run!" << std::endl;
      }

      if ( runType_ != sistrip::APV_TIMING ) { 
	ss << "  APV timing maj/min vers    : " << apvTimingV_.first << "." << apvTimingV_.second << std::endl;
      } else {
	ss << "  APV timing maj/min vers    : " << runTableVersion_.first << "." << runTableVersion_.second
	   << " <= This \"state\" version overriden by \"history\" version " 
	   << apvTimingV_.first << "." << apvTimingV_.second 
	   << " for this APV timing run!" << std::endl;
      }

      if ( runType_ != sistrip::OPTO_SCAN ) { 
	ss << "  Opto scan maj/min vers     : " << optoScanV_.first << "." << optoScanV_.second << std::endl;
      } else {
	ss << "  Opto scan maj/min vers     : " << runTableVersion_.first << "." << runTableVersion_.second
	   << " <= This \"state\" version overriden by \"history\" version " 
	   << optoScanV_.first << "." << optoScanV_.second 
	   << " for this opto scan run!" << std::endl;
      }

      if ( runType_ != sistrip::VPSP_SCAN ) { 
	ss << "  VPSP scan maj/min vers     : " << vpspScanV_.first << "." << vpspScanV_.second << std::endl;
      } else {
	ss << "  VPSP scan maj/min vers     : " << runTableVersion_.first << "." << runTableVersion_.second
	   << " <= This \"state\" version overriden by \"history\" version " 
	   << vpspScanV_.first << "." << vpspScanV_.second 
	   << " for this VPSP scan run!" << std::endl;
      }

      if ( runType_ != sistrip::CALIBRATION ) { 
	ss << "  APV calib maj/min vers     : " << apvCalibV_.first << "." << apvCalibV_.second << std::endl;
      } else {
	ss << "  APV calib maj/min vers     : " << runTableVersion_.first << "." << runTableVersion_.second
	   << " <= This \"state\" version overriden by \"history\" version " 
	   << apvCalibV_.first << "." << apvCalibV_.second 
	   << " for this APV calib run!" << std::endl;
      }

      if ( runType_ != sistrip::PEDESTALS ) { 
	ss << "  Pedestals maj/min vers     : " << pedestalsV_.first << "." << pedestalsV_.second << std::endl;
      } else {
	ss << "  Pedestals maj/min vers     : " << runTableVersion_.first << "." << runTableVersion_.second
	   << " <= This \"state\" version overriden by \"history\" version " 
	   << pedestalsV_.first << "." << pedestalsV_.second 
	   << " for this pedestals run!" << std::endl;
      }

      if ( runType_ != sistrip::APV_LATENCY ) { 
	ss << "  APV latency maj/min vers   : " << apvLatencyV_.first << "." << apvLatencyV_.second << std::endl;
      } else {
	ss << "  APV latency maj/min vers   : " << runTableVersion_.first << "." << runTableVersion_.second
	   << " <= This \"state\" version overriden by \"history\" version " 
	   << apvLatencyV_.first << "." << apvLatencyV_.second 
	   << " for this APV latency run!" << std::endl;
      }

      if ( runType_ != sistrip::FINE_DELAY_TTC ) { 
	ss << "  Fine delay maj/min vers    : " << fineDelayV_.first << "." << fineDelayV_.second << std::endl;
      } else {
	ss << "  Fine delay maj/min vers    : " << runTableVersion_.first << "." << runTableVersion_.second
	   << " <= This \"state\" version overriden by \"history\" version " 
	   << fineDelayV_.first << "." << fineDelayV_.second 
	   << " for this fine delay run!" << std::endl;
      }

    }

  } else {
    
    ss << "  Input \"module.xml\" file    : " << inputModuleXml_ << std::endl
       << "  Input \"dcuinfo.xml\" file   : " << inputDcuInfoXml_ << std::endl
       << "  Input \"fec.xml\" file(s)    : ";
    std::vector<std::string>::const_iterator ifec = inputFecXml_.begin();
    for ( ; ifec != inputFecXml_.end(); ifec++ ) { ss << *ifec << ", "; }
    ss << std::endl;
    ss << "  Input \"fed.xml\" file(s)    : ";
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
SiStripPartition::Versions SiStripPartition::versions( std::vector<uint32_t> input ) {
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
