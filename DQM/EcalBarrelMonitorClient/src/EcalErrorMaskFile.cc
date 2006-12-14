// $Id: EcalErrorMaskFile.cc,v 1.2 2006/12/14 14:16:14 dellaric Exp $

/*!
  \file EcalErrorMaskFile.cc
  \brief Error mask from text file
  \author B. Gobbo 
  \version $Revision: 1.2 $
  \date $Date: 2006/12/14 14:16:14 $
*/

#include "DQM/EcalBarrelMonitorClient/interface/EcalErrorMaskFile.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex.h>
#include <CondFormats/EcalObjects/interface/EcalErrorDictionary.h>

bool EcalErrorMaskFile::done_ = false;
std::string EcalErrorMaskFile::inFile_ = "";
std::map<EcalLogicID, MonCrystalStatusDat> EcalErrorMaskFile::mapMCSD_;
std::map<EcalLogicID, MonPNStatusDat>      EcalErrorMaskFile::mapMPSD_;

void EcalErrorMaskFile::readFile( std::string inFile ) throw( std::runtime_error ) {

  if( done_ ) {
      throw( std::runtime_error( "Input File already read." ) );
      return;
  }

  done_ = true;

  const unsigned int lineSize = 512;
  char line[lineSize];

  std::fstream f( inFile.c_str(), std::ios::in );
  if( f.fail() ) {
    std::string s = "Error accessing input file " + inFile;
    throw( std::runtime_error( s ) );
    return;
  }

  int linecount = 0;

  // Local copy of error dictionary
  std::vector<EcalErrorDictionary::errorDef_t> errors;
  EcalErrorDictionary::getDictionary( errors );

  while( f.getline( line, lineSize ) ) {

    linecount++;

    EcalErrorMaskFile::clearComments_( line );
    EcalErrorMaskFile::clearFinalBlanks_( line );
  
    std::istringstream is( line );
    std::string s;
    is >> s;
    if( s == "MonCrystalStatusDat" ) {
      int sm; is >> sm;
      if( sm < 1 || sm > 36 ) {
	std::ostringstream os;
	os << "line " << linecount << ": SM must be a number between 1 and 36" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      int ic; is >> ic;
      if( ic < 1 || ic > 1700 ) {
	std::ostringstream os;
	os << "line " << linecount << ": IC must be a number between 1 and 1700" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      unsigned char gain;
      std::string ga; is >> ga;
      gain = 0;
      if( ga == "*" ) gain = 0x111;
      else if( ga ==  "1" ) gain = 0x001;
      else if( ga ==  "6" ) gain = 0x010;
      else if( ga == "12" ) gain = 0x100;
      else {
	std::ostringstream os;
	os << "line " << linecount << ": GAIN must be 1, 6, 12 or *" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      std::string shortDesc; is >> shortDesc;
      uint64_t bitmask; bitmask = 0;
      std::string longDesc; longDesc = "";
      
      for( unsigned int i=0; i<errors.size(); i++ ) {
	if( shortDesc == errors[i].shortDesc ) {
	  bitmask = errors[i].bitmask;
	  longDesc = errors[i].longDesc;
	}
      }
      if( bitmask == 0 ) {
	std::ostringstream os;
	os << "line " << linecount << ": This Short Description was not fount in the Dictionary" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      pair<EcalLogicID, MonCrystalStatusDat> pMCSD;
      MonCrystalStatusDef mcsf;
      mcsf.setShortDesc( shortDesc );
      MonCrystalStatusDat mcsd;
      if( gain & 0x001 ) mcsd.setStatusG1( mcsf ); 
      if( gain & 0x010 ) mcsd.setStatusG6( mcsf ); 
      if( gain & 0x100 ) mcsd.setStatusG12( mcsf ); 
      pMCSD.first = EcalLogicID( "local", 10000*(sm-1)+ic );
      pMCSD.second = mcsd;
      EcalErrorMaskFile::mapMCSD_.insert( pMCSD );
    }
    else if( s == "MonPNStatusDat" ) {
      throw( std::runtime_error( "MonPNStatusDat: To be implemented" ) );
    }
    else if( s == "MonMemChStatusDat" ) {
      throw( std::runtime_error( "MonMemChStatusDat: To be implemented" ) );
    }
    else if( s == "MonMemTTStatusDat" ) {
      throw( std::runtime_error( "MonMemTTStatusDat: To be implemented" ) );
    }
    else {
      throw( std::runtime_error( "Wrong Table Name" ) );
    }

  }

  return;

}

template <class T> void EcalErrorMaskFile::fetchData( std::map< EcalLogicID, T>& fillMap ) throw( std::runtime_error ) {

  if( !done_ ) {
    throw( std::runtime_error( "Input file not read" ) );
    return;
  }
  fillMap->clear();
  if( dynamic_cast<MonCrystalStatusDat*>(&(fillMap.second)) ) {
    *fillMap = EcalErrorMaskFile::mapMCSD_;
    return;
  }
  else if( dynamic_cast<MonPNStatusDat*>(&(fillMap.second)) ) {
    *fillMap = EcalErrorMaskFile::mapMPSD_;
    return;
  }
  else {
    throw( std::runtime_error( "Table unknown or not yet implemented." ) );
  }

}

void EcalErrorMaskFile::clearComments_( char* line ) {
  // It looks for "#" and replaces it with "\0"...
  regex_t rec;
  regmatch_t pmc;

  (void) regcomp( &rec, "#", REG_EXTENDED );

  int i = regexec( &rec, line, (size_t) 1, &pmc, 0 ); 

  if( i == 0 ) {
    line[pmc.rm_so] = '\0';
  }

  regfree( &rec );
}

void EcalErrorMaskFile::clearFinalBlanks_( char* line ) {
  // From end of string, find last ' ' or '\t' (tab) and replece it with '\0'
  int i;
  for( i=strlen(line)-1; i>=0 && (line[i]==' '||line[i]=='\t'); i-- );
  if( line[i+1]  == ' ' || line[i+1] == '\t' ) line[i+1] = '\0';
}
