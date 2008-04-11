// Last commit: $Id: $

#include "OnlineDB/SiStripConfigDb/interface/SiStripPartition.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripPartition::SiStripPartition() :
  partitionName_(""), 
  runNumber_(0),
  runType_(sistrip::UNDEFINED_RUN_TYPE),
  cabMajor_(0),
  cabMinor_(0),
  fedMajor_(0),
  fedMinor_(0),
  fecMajor_(0),
  fecMinor_(0),
  calMajor_(0),
  calMinor_(0),
  dcuMajor_(0),
  dcuMinor_(0),
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

  cabMajor_  = 0;
  cabMinor_  = 0;
  fedMajor_  = 0;
  fedMinor_  = 0;
  fecMajor_  = 0;
  fecMinor_  = 0;
  calMajor_  = 0;
  calMinor_  = 0;
  dcuMajor_  = 0;
  dcuMinor_  = 0;

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

  cabMajor_  = pset.getUntrackedParameter<unsigned int>( "CablingMajorVersion", 0 );
  cabMinor_  = pset.getUntrackedParameter<unsigned int>( "CablingMinorVersion", 0 );
  fedMajor_  = pset.getUntrackedParameter<unsigned int>( "FedMajorVersion", 0 );
  fedMinor_  = pset.getUntrackedParameter<unsigned int>( "FedMinorVersion", 0 );
  fecMajor_  = pset.getUntrackedParameter<unsigned int>( "FecMajorVersion", 0 );
  fecMinor_  = pset.getUntrackedParameter<unsigned int>( "FecMinorVersion", 0 );
  dcuMajor_  = pset.getUntrackedParameter<unsigned int>( "DcuDetIdMajorVersion", 0 );
  dcuMinor_  = pset.getUntrackedParameter<unsigned int>( "DcuDetIdMinorVersion", 0 );
  calMajor_  = pset.getUntrackedParameter<unsigned int>( "CalibMajorVersion", 0 );
  calMinor_  = pset.getUntrackedParameter<unsigned int>( "CalibMinorVersion", 0 );

  std::vector<std::string> tmp(1,"");
  inputModuleXml_   = pset.getUntrackedParameter<std::string>( "InputModuleXml", "" );
  inputDcuInfoXml_  = pset.getUntrackedParameter<std::string>( "InputDcuInfoXml", "" ); 
  inputFecXml_      = pset.getUntrackedParameter< std::vector<std::string> >( "InputFecXml", tmp ); 
  inputFedXml_      = pset.getUntrackedParameter< std::vector<std::string> >( "InputFedXml", tmp );

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
    
    ss << "  Cabling major/minor vers  : " << cabMajor_ << "." << cabMinor_ << std::endl
       << "  FED major/minor vers      : " << fedMajor_ << "." << fedMinor_ << std::endl
       << "  FEC major/minor vers      : " << fecMajor_ << "." << fecMinor_ << std::endl
       << "  Calibration maj/min vers  : " << calMajor_ << "." << calMinor_ << std::endl
       << "  DCU-DetId maj/min vers    : " << dcuMajor_ << "." << dcuMinor_ << std::endl;
    
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
