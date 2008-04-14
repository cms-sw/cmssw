// Last commit: $Id: SiStripPartition.cc,v 1.1 2008/04/11 13:27:33 bainbrid Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripPartition.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
  forceVersions_(true),
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
  partitionName_ = "";
  runNumber_     = 0;
  runType_       = sistrip::UNDEFINED_RUN_TYPE;
  forceVersions_ = true;

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
void SiStripPartition::setParams( const edm::ParameterSet& pset ) {

  partitionName_ = pset.getUntrackedParameter<std::string>( "PartitionName", "" );
  runNumber_     = pset.getUntrackedParameter<unsigned int>( "RunNumber", 0 );
  forceVersions_ = pset.getUntrackedParameter<bool>( "ForceVersions", false );

  cabVersion_ = versions( pset.getUntrackedParameter< std::vector<unsigned int> >( "CablingVersion", std::vector<unsigned int>(2,0) ) );
  fedVersion_ = versions( pset.getUntrackedParameter< std::vector<unsigned int> >( "FedVersion", std::vector<unsigned int>(2,0) ) );
  fecVersion_ = versions( pset.getUntrackedParameter< std::vector<unsigned int> >( "FecVersion", std::vector<unsigned int>(2,0) ) );
  dcuVersion_ = versions( pset.getUntrackedParameter< std::vector<unsigned int> >( "DcuDetIdVersion", std::vector<unsigned int>(2,0) ) );
  calVersion_ = versions( pset.getUntrackedParameter< std::vector<unsigned int> >( "CalibVersion", std::vector<unsigned int>(2,0) ) );
  
  std::vector<std::string> tmp(1,"");
  inputModuleXml_   = pset.getUntrackedParameter<std::string>( "InputModuleXml", "" );
  inputDcuInfoXml_  = pset.getUntrackedParameter<std::string>( "InputDcuInfoXml", "" ); 
  inputFecXml_      = pset.getUntrackedParameter< std::vector<std::string> >( "InputFecXml", tmp ); 
  inputFedXml_      = pset.getUntrackedParameter< std::vector<std::string> >( "InputFedXml", tmp );

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

// -----------------------------------------------------------------------------
// 
void SiStripPartition::print( std::stringstream& ss, bool using_db ) const {

  ss << "  Partition                 : " << partitionName_ << std::endl;
  
  if ( using_db ) {
    
    ss << "  Force versions            : " << std::boolalpha << forceVersions_ << std::noboolalpha << std::endl;
    ss << "  Run number                : ";
    if ( !forceVersions_ ) { ss << runNumber_; }
    else { ss << "(Overriden by versions specified below!)"; }
    ss << std::endl;
    if ( !forceVersions_ ) { 
      ss << "  Run type                  : " << SiStripEnumsAndStrings::runType( runType_ ) << std::endl;
    }
    
    ss << "  Cabling major/minor vers  : " << cabVersion_.first << "." << cabVersion_.second << std::endl
       << "  FED major/minor vers      : " << fedVersion_.first << "." << fedVersion_.second << std::endl
       << "  FEC major/minor vers      : " << fecVersion_.first << "." << fecVersion_.second << std::endl
       << "  Calibration maj/min vers  : " << calVersion_.first << "." << calVersion_.second << std::endl
       << "  DCU-DetId maj/min vers    : " << dcuVersion_.first << "." << dcuVersion_.second << std::endl;
    
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
