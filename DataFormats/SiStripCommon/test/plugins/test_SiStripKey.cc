// Last commit: $Id: test_SiStripKey.cc,v 1.5 2010/01/07 11:21:03 lowette Exp $

#include "DataFormats/SiStripCommon/test/plugins/test_SiStripKey.h"
#include "FWCore/Framework/interface/Event.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripDetKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <time.h>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
testSiStripKey::testSiStripKey( const edm::ParameterSet& pset ) 
  : keyType_( sistrip::UNKNOWN_KEY ),
    key_(0),
    path_( pset.getUntrackedParameter<std::string>("Path","") )
{
  
  LogTrace(mlDqmCommon_)
    << "[testSiStripKey::" << __func__ << "]"
    << " Constructing object...";
  
  // extract key type
  std::string key_type = pset.getUntrackedParameter<std::string>("KeyType","");
  keyType_ = SiStripEnumsAndStrings::keyType( key_type );
  
  // extract key without hex prefix
  std::stringstream key;
  std::string tmp = pset.getUntrackedParameter<std::string>("Key","0x0");
  if ( tmp.find( sistrip::hex_ ) != std::string::npos ) {
    key << std::string( tmp, (sizeof(sistrip::hex_) - 1), tmp.size() ); 
  } else { key << tmp; }
  key >> std::hex >> key_;
  
}

// -----------------------------------------------------------------------------
// 
testSiStripKey::~testSiStripKey() {
  LogTrace(mlDqmCommon_)
    << "[testSiStripKey::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
void testSiStripKey::beginJob() {
  
  edm::LogVerbatim(mlDqmCommon_)
    << "[SiStripKey::" << __func__ << "]"
    << " Tests the generation of keys...";
  
  if ( keyType_ == sistrip::FED_KEY ) {

    SiStripFedKey from_key( key_ );
    edm::LogVerbatim(mlDqmCommon_)
      << "[SiStripKey::" << __func__ << "]"
      << " FED key object built from 32-bit key: "
      << std::endl << from_key;

    SiStripFedKey from_path( path_ );
    edm::LogVerbatim(mlDqmCommon_)
      << "[SiStripKey::" << __func__ << "]"
      << " FED key object built from directory string: " 
      << std::endl << from_path;
    
  } else if ( keyType_ == sistrip::FEC_KEY ) { 

    SiStripFecKey from_key( key_ );
    edm::LogVerbatim(mlDqmCommon_)
      << "[SiStripKey::" << __func__ << "]"
      << " FEC key object built from 32-bit key: " 
      << std::endl << from_key;

    SiStripFecKey from_path( path_ );
    edm::LogVerbatim(mlDqmCommon_)
      << "[SiStripKey::" << __func__ << "]"
      << " FEC key object built from directory string: " 
      << std::endl << from_path;
    
  } else if ( keyType_ == sistrip::DET_KEY ) { 

    SiStripDetKey from_key( key_ );
    edm::LogVerbatim(mlDqmCommon_)
      << "[SiStripKey::" << __func__ << "]"
      << " DET key object built from 32-bit key: " 
      << std::endl << from_key;

    SiStripDetKey from_path( path_ );
    edm::LogVerbatim(mlDqmCommon_)
      << "[SiStripKey::" << __func__ << "]"
      << " DET key object built from directory string: " 
      << std::endl << from_path;

  } else {

    // warning
    edm::LogWarning(mlDqmCommon_)
      << "[SiStripKey::" << __func__ << "]"
      << " KeyType is of type: " 
      << SiStripEnumsAndStrings::keyType( keyType_ );

    // fed 
    {
      SiStripFedKey from_key( key_ );
      edm::LogVerbatim(mlDqmCommon_)
	<< "[SiStripKey::" << __func__ << "]"
	<< " FED key object built from 32-bit key: " 
	<< std::endl << from_key;

      SiStripFedKey from_path( path_ );
      edm::LogVerbatim(mlDqmCommon_)
	<< "[SiStripKey::" << __func__ << "]"
	<< " FED key object built from directory string: " 
	<< std::endl << from_path;
    }

    // fec
    {
      SiStripFecKey from_key( key_ );
      edm::LogVerbatim(mlDqmCommon_)
	<< "[SiStripKey::" << __func__ << "]"
	<< " FEC key object built from 32-bit key: " 
	<< std::endl << from_key;

      SiStripFecKey from_path( path_ );
      edm::LogVerbatim(mlDqmCommon_)
	<< "[SiStripKey::" << __func__ << "]"
	<< " FEC key object built from directory string: " 
	<< std::endl << from_path;
    }

    // det
    {
      SiStripDetKey from_key( key_ );
      edm::LogVerbatim(mlDqmCommon_)
	<< "[SiStripKey::" << __func__ << "]"
	<< " DET key object built from 32-bit key: " 
	<< std::endl << from_key;

      SiStripDetKey from_path( path_ );
      edm::LogVerbatim(mlDqmCommon_)
	<< "[SiStripKey::" << __func__ << "]"
	<< " DET key object built from directory string: " 
	<< std::endl << from_path;
    }

  }

}

// -----------------------------------------------------------------------------
// 
void testSiStripKey::analyze( const edm::Event& event, 
			       const edm::EventSetup& setup ) {
  LogTrace(mlDqmCommon_) 
    << "[SiStripKey::" << __func__ << "]"
    << " Analyzing run/event "
    << event.id().run() << "/"
    << event.id().event();
}


