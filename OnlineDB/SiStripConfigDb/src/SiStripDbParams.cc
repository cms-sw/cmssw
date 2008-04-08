// Last commit: $Id: $

#include "OnlineDB/SiStripConfigDb/interface/SiStripDbParams.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripDbParams::SiStripDbParams() :
  usingDb_(false),
  user_(null_),
  passwd_(null_),
  path_(null_),
  partitions_(), 
  usingDbCache_(false),
  sharedMemory_(""),
  runNumber_(0),
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
  runType_(sistrip::UNDEFINED_RUN_TYPE),
  force_(true),
  inputModuleXml_(""),
  inputDcuInfoXml_(""),
  inputFecXml_(),
  inputFedXml_(),
  inputDcuConvXml_(""),
  outputModuleXml_("/tmp/module.xml"),
  outputDcuInfoXml_("/tmp/dcuinfo.xml"),
  outputFecXml_("/tmp/fec.xml"),
  outputFedXml_("/tmp/fed.xml"),
  tnsAdmin_("")
{
  reset();
}

// -----------------------------------------------------------------------------
// 
SiStripDbParams::~SiStripDbParams() {
  inputFecXml_.clear();
  inputFedXml_.clear();
}

// -----------------------------------------------------------------------------
// 
void SiStripDbParams::reset() {
  usingDb_ = false;
  confdb_ = null_;
  confdb( confdb_ );
  partitions_ = std::vector<std::string>();
  usingDbCache_ = false;
  sharedMemory_ = "";
  runNumber_ = 0;
  cabMajor_ = 0;
  cabMinor_ = 0;
  fedMajor_ = 0;
  fedMinor_ = 0;
  fecMajor_ = 0;
  fecMinor_ = 0;
  calMajor_ = 0;
  calMinor_ = 0;
  dcuMajor_ = 0;
  dcuMinor_ = 0;
  runType_  = sistrip::UNDEFINED_RUN_TYPE;
  force_    = true;
  inputModuleXml_   = "";
  inputDcuInfoXml_  = "";
  inputFecXml_      = std::vector<std::string>(1,"");
  inputFedXml_      = std::vector<std::string>(1,"");
  inputDcuConvXml_  = "";
  outputModuleXml_  = "";
  outputDcuInfoXml_ = "";
  outputFecXml_     = "";
  outputFedXml_     = "";
  tnsAdmin_         = "";
}

// -----------------------------------------------------------------------------
// 
void SiStripDbParams::setParams( const edm::ParameterSet& pset ) {
  reset();
  usingDb_ = pset.getUntrackedParameter<bool>( "UsingDb", false ); 
  confdb( pset.getUntrackedParameter<std::string>( "ConfDb", "") );
  partitions_ = pset.getUntrackedParameter< std::vector<std::string> >( "Partitions", std::vector<std::string>() );
  runNumber_ = pset.getUntrackedParameter<unsigned int>( "RunNumber", 0 );
  usingDbCache_ = pset.getUntrackedParameter<bool>( "UsingDbCache", false ); 
  sharedMemory_ = pset.getUntrackedParameter<std::string>( "SharedMemory", "" ); 
  cabMajor_ = pset.getUntrackedParameter<unsigned int>( "MajorVersion", 0 );
  cabMinor_ = pset.getUntrackedParameter<unsigned int>( "MinorVersion", 0 );
  fedMajor_ = pset.getUntrackedParameter<unsigned int>( "FedMajorVersion", 0 );
  fedMinor_ = pset.getUntrackedParameter<unsigned int>( "FedMinorVersion", 0 );
  fecMajor_ = pset.getUntrackedParameter<unsigned int>( "FecMajorVersion", 0 );
  fecMinor_ = pset.getUntrackedParameter<unsigned int>( "FecMinorVersion", 0 );
  dcuMajor_ = pset.getUntrackedParameter<unsigned int>( "DcuDetIdMajorVersion", 0 );
  dcuMinor_ = pset.getUntrackedParameter<unsigned int>( "DcuDetIdMinorVersion", 0 );
  calMajor_ = pset.getUntrackedParameter<unsigned int>( "CalibMajorVersion", 0 );
  calMinor_ = pset.getUntrackedParameter<unsigned int>( "CalibMinorVersion", 0 );
  force_ = pset.getUntrackedParameter<bool>( "ForceDcuDetIdVersions", true );
  inputModuleXml_ = pset.getUntrackedParameter<std::string>( "InputModuleXml", "" );
  inputDcuInfoXml_ = pset.getUntrackedParameter<std::string>( "InputDcuInfoXml", "" ); 
  inputFecXml_ = pset.getUntrackedParameter< std::vector<std::string> >( "InputFecXml", std::vector<std::string>(1,"") ); 
  inputFedXml_ = pset.getUntrackedParameter< std::vector<std::string> >( "InputFedXml", std::vector<std::string>(1,"") ); 
  inputDcuConvXml_ = pset.getUntrackedParameter<std::string>( "InputDcuConvXml", "" );
  outputModuleXml_ = pset.getUntrackedParameter<std::string>( "OutputModuleXml", "/tmp/module.xml" );
  outputDcuInfoXml_ = pset.getUntrackedParameter<std::string>( "OutputDcuInfoXml", "/tmp/dcuinfo.xml" );
  outputFecXml_ = pset.getUntrackedParameter<std::string>( "OutputFecXml", "/tmp/fec.xml" );
  outputFedXml_ = pset.getUntrackedParameter<std::string>( "OutputFedXml", "/tmp/fed.xml" );
  tnsAdmin_ = pset.getUntrackedParameter<std::string>( "TNS_ADMIN", "" );
}

// -----------------------------------------------------------------------------
// 
void SiStripDbParams::confdb( const std::string& confdb ) {
  confdb_ = confdb;
  uint32_t ipass = confdb.find("/");
  uint32_t ipath = confdb.find("@");
  if ( ipass != std::string::npos && 
       ipath != std::string::npos ) {
    user_   = confdb.substr(0,ipass); 
    passwd_ = confdb.substr(ipass+1,ipath-ipass-1); 
    path_   = confdb.substr(ipath+1,confdb.size());
  } else {
    user_   = null_;
    passwd_ = null_;
    path_   = null_;
  }
}

// -----------------------------------------------------------------------------
// 
void SiStripDbParams::confdb( const std::string& user,
					const std::string& passwd,
					const std::string& path ) {
  if ( user != "" && passwd != "" && path != "" &&
       user != null_ && passwd != null_ && path != null_ ) {
    user_   = user;
    passwd_ = passwd;
    path_   = path;
  } else {
    user_   = null_;
    passwd_ = null_;
    path_   = null_;
  }
  confdb_ = user_ + "/" + passwd_ + "@" + path_;
}

// -----------------------------------------------------------------------------
// 
std::string SiStripDbParams::partitions() const {
  std::stringstream ss;
  std::vector<std::string>::const_iterator ii = partitions_.begin();
  std::vector<std::string>::const_iterator jj = partitions_.end();
  for ( ; ii != jj; ++ii ) { ii == partitions_.begin() ? ss << *ii : ss << ", " << *ii; }
  return ss.str();
}

// -----------------------------------------------------------------------------
// 
std::vector<std::string> SiStripDbParams::partitions( std::string input ) const {
  std::istringstream ss(input);
  std::vector<std::string> partitions;
  std::string delimiter = ":";
  std::string token;
  while ( getline( ss, token, ':' ) ) { partitions.push_back(token); }
  return partitions;
}

// -----------------------------------------------------------------------------
// 
void SiStripDbParams::print( std::stringstream& ss ) const {

  ss << " Using database account    : " << std::boolalpha << usingDb_ << std::noboolalpha << std::endl;
  ss << " Using database cache      : " << std::boolalpha << usingDbCache_ << std::noboolalpha << std::endl;
  ss << " Shared memory name        : " << std::boolalpha << sharedMemory_ << std::noboolalpha << std::endl;

  if ( usingDb_ ) {

    ss << " ConfDb                    : " << confdb_ << std::endl;
      //<< " User, Passwd, Path        : " << user_ << ", " << passwd_ << ", " << path_ << std::endl;

  } else {

    // Input
    ss << " Input \"module.xml\" file   : " << inputModuleXml_ << std::endl
       << " Input \"dcuinfo.xml\" file  : " << inputDcuInfoXml_ << std::endl
       << " Input \"fec.xml\" file(s)   : ";
    std::vector<std::string>::const_iterator ifec = inputFecXml_.begin();
    for ( ; ifec != inputFecXml_.end(); ifec++ ) { ss << *ifec << ", "; }
    ss << std::endl;
    ss << " Input \"fed.xml\" file(s)   : ";
    std::vector<std::string>::const_iterator ifed = inputFedXml_.begin();
    for ( ; ifed != inputFedXml_.end(); ifed++ ) { ss << *ifed << ", "; }
    ss << std::endl;

    // Output 
    ss << " Output \"module.xml\" file  : " << outputModuleXml_ << std::endl
       << " Output \"dcuinfo.xml\" file : " << outputDcuInfoXml_ << std::endl
       << " Output \"fec.xml\" file(s)  : " << outputFecXml_ << std::endl
       << " Output \"fed.xml\" file(s)  : " << outputFedXml_ << std::endl;

  }
  
  ss << " Partitions                : " << partitions() << std::endl;
  ss << " Run number                : " << runNumber_ << std::endl
     << " Run type                  : " << SiStripEnumsAndStrings::runType( runType_ ) << std::endl
     << " Cabling major/minor vers  : " << cabMajor_ << "." << cabMinor_ << std::endl
     << " FED major/minor vers      : " << fedMajor_ << "." << fedMinor_ << std::endl
     << " FEC major/minor vers      : " << fecMajor_ << "." << fecMinor_ << std::endl
     << " Calibration maj/min vers  : " << calMajor_ << "." << calMinor_ << std::endl
     << " DCU-DetId maj/min vers    : " << dcuMajor_ << "." << dcuMinor_;
  if ( force_ ) { ss << " (version not overridden by run number)"; }
  ss << std::endl;
  
}

// -----------------------------------------------------------------------------
// 
std::ostream& operator<< ( std::ostream& os, const SiStripDbParams& params ) {
  std::stringstream ss;
  params.print(ss);
  os << ss.str();
  return os;
}
