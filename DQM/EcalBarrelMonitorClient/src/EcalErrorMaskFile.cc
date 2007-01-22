// $Id: EcalErrorMaskFile.cc,v 1.17 2007/01/21 17:40:53 dellaric Exp $

/*!
  \file EcalErrorMaskFile.cc
  \brief Error mask from text file
  \author B. Gobbo 
  \version $Revision: 1.17 $
  \date $Date: 2007/01/21 17:40:53 $
*/

#include "DQM/EcalBarrelMonitorClient/interface/EcalErrorMaskFile.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex.h>
#include <CondTools/Ecal/interface/EcalErrorDictionary.h>

bool EcalErrorMaskFile::done_ = false;
std::string EcalErrorMaskFile::inFile_ = "";
std::map<EcalLogicID, RunCrystalErrorsDat> EcalErrorMaskFile::mapCrystalErrors_;
std::map<EcalLogicID, RunTTErrorsDat>      EcalErrorMaskFile::mapTTErrors_;
std::map<EcalLogicID, RunPNErrorsDat>      EcalErrorMaskFile::mapPNErrors_;
std::map<EcalLogicID, RunMemChErrorsDat>   EcalErrorMaskFile::mapMemChErrors_;
std::map<EcalLogicID, RunMemTTErrorsDat>   EcalErrorMaskFile::mapMemTTErrors_;

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
    if( s == "" ) continue;

    int sm; is >> sm;
    if( sm < 1 || sm > 36 ) {
      std::ostringstream os;
      os << "line " << linecount << ": SM must be a number between 1 and 36" << std::ends;
      throw( std::runtime_error( os.str() ) );
      return;
    }
    
    if( s == "Crystal" ) {
      int ic; is >> ic;
      if( ic < 1 || ic > 1700 ) {
	std::ostringstream os;
	os << "line " << linecount << ": IC must be a number between 1 and 1700" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      std::string shortDesc; is >> shortDesc;
      uint64_t bitmask; bitmask = 0;
      
      for( unsigned int i=0; i<errors.size(); i++ ) {
	if( shortDesc == errors[i].shortDesc ) {
	  bitmask = errors[i].bitmask;
	}
      }
      if( bitmask == 0 ) {
	std::ostringstream os;
	os << "line " << linecount << ": This Short Description was not found in the Dictionary" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      EcalLogicID id = EcalLogicID( "local", 10000*sm+ic, sm, ic, 0 );
      std::map<EcalLogicID, RunCrystalErrorsDat>::iterator i = EcalErrorMaskFile::mapCrystalErrors_.find( id );
      if( i != mapCrystalErrors_.end() ) {
	uint64_t oldBitmask = (i->second).getErrorBits();
	oldBitmask |= bitmask;
	(i->second).setErrorBits( oldBitmask );
      }
      else {
	RunCrystalErrorsDat error;
	error.setErrorBits(bitmask);
	EcalErrorMaskFile::mapCrystalErrors_[ id ] = error;
      }
    }
    else if( s == "TT" ) {
      int it; is >> it;
      if( it < 1 || it > 68 ) {
	std::ostringstream os;
	os << "line " << linecount << ": IT must be a number between 1 and 68" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      std::string shortDesc; is >> shortDesc;
      uint64_t bitmask; bitmask = 0;
      
      for( unsigned int i=0; i<errors.size(); i++ ) {
	if( shortDesc == errors[i].shortDesc ) {
	  bitmask = errors[i].bitmask;
	}
      }
      if( bitmask == 0 ) {
	std::ostringstream os;
	os << "line " << linecount << ": This Short Description was not found in the Dictionary" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      EcalLogicID id = EcalLogicID( "local", 10000*sm+it, sm, it, 0 );
      std::map<EcalLogicID, RunTTErrorsDat>::iterator i = EcalErrorMaskFile::mapTTErrors_.find( id );
      if( i != mapTTErrors_.end() ) {
	uint64_t oldBitmask = (i->second).getErrorBits();
	oldBitmask |= bitmask;
	(i->second).setErrorBits( oldBitmask );
      }
      else {
	RunTTErrorsDat error;
	error.setErrorBits(bitmask);
	EcalErrorMaskFile::mapTTErrors_[ id ] = error;
      }
    }
    else if( s == "PN" ) {
      int ic; is >> ic;
      if( ic < 1 || ic > 10 ) {
	std::ostringstream os;
	os << "line " << linecount << ": IC must be a number between 1 and 10" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      std::string shortDesc; is >> shortDesc;
      uint64_t bitmask; bitmask = 0;
      
      for( unsigned int i=0; i<errors.size(); i++ ) {
	if( shortDesc == errors[i].shortDesc ) {
	  bitmask = errors[i].bitmask;
	}
      }
      if( bitmask == 0 ) {
	std::ostringstream os;
	os << "line " << linecount << ": This Short Description was not found in the Dictionary" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      EcalLogicID id = EcalLogicID( "local", 10000*sm+ic, sm, ic, 0 );
      std::map<EcalLogicID, RunPNErrorsDat>::iterator i = EcalErrorMaskFile::mapPNErrors_.find( id );
      if( i != mapPNErrors_.end() ) {
	uint64_t oldBitmask = (i->second).getErrorBits();
	oldBitmask |= bitmask;
	(i->second).setErrorBits( oldBitmask );
      }
      else {
	RunPNErrorsDat error;
	error.setErrorBits(bitmask);
	EcalErrorMaskFile::mapPNErrors_[ id ] = error;
      }
    }
    else if( s == "MemCh" ) {
      int ic; is >> ic;
      if( ic < 1 || ic > 50 ) {
	std::ostringstream os;
	os << "line " << linecount << ": IC must be a number between 1 and 50" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      std::string shortDesc; is >> shortDesc;
      uint64_t bitmask; bitmask = 0;
      
      for( unsigned int i=0; i<errors.size(); i++ ) {
	if( shortDesc == errors[i].shortDesc ) {
	  bitmask = errors[i].bitmask;
	}
      }
      if( bitmask == 0 ) {
	std::ostringstream os;
	os << "line " << linecount << ": This Short Description was not found in the Dictionary" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      EcalLogicID id = EcalLogicID( "local", 10000*sm+ic, sm, ic, 0 );
      std::map<EcalLogicID, RunMemChErrorsDat>::iterator i = EcalErrorMaskFile::mapMemChErrors_.find( id );
      if( i != mapMemChErrors_.end() ) {
	uint64_t oldBitmask = (i->second).getErrorBits();
	oldBitmask |= bitmask;
	(i->second).setErrorBits( oldBitmask );
      }
      else {
	RunMemChErrorsDat error;
	error.setErrorBits(bitmask);
	EcalErrorMaskFile::mapMemChErrors_[ id ] = error;
      }
    }
    else if( s == "MemTT" ) {
      int it; is >> it;
      if( it < 69 || it > 70 ) {
	std::ostringstream os;
	os << "line " << linecount << ": IT must be 69 or 70" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      std::string shortDesc; is >> shortDesc;
      uint64_t bitmask; bitmask = 0;
      
      for( unsigned int i=0; i<errors.size(); i++ ) {
	if( shortDesc == errors[i].shortDesc ) {
	  bitmask = errors[i].bitmask;
	}
      }
      if( bitmask == 0 ) {
	std::ostringstream os;
	os << "line " << linecount << ": This Short Description was not found in the Dictionary" << std::ends;
	throw( std::runtime_error( os.str() ) );
	return;
      }
      EcalLogicID id = EcalLogicID( "local", 10000*sm+it, sm, it, 0 );
      std::map<EcalLogicID, RunMemTTErrorsDat>::iterator i = EcalErrorMaskFile::mapMemTTErrors_.find( id );
      if( i != mapMemTTErrors_.end() ) {
	uint64_t oldBitmask = (i->second).getErrorBits();
	oldBitmask |= bitmask;
	(i->second).setErrorBits( oldBitmask );
      }
      else {
	RunMemTTErrorsDat error;
	error.setErrorBits(bitmask);
	EcalErrorMaskFile::mapMemTTErrors_[ id ] = error;
      }
    }
    else {
      throw( std::runtime_error( "Wrong Table Name" ) );
    }

  }

  return;

}

void EcalErrorMaskFile::fetchDataSet( std::map< EcalLogicID, RunCrystalErrorsDat>* fillMap ) throw( std::runtime_error ) {

  if( !done_ ) {
    throw( std::runtime_error( "Input file not read" ) );
    return;
  }
  fillMap->clear();
  *fillMap = EcalErrorMaskFile::mapCrystalErrors_;
  return;
}

void EcalErrorMaskFile::fetchDataSet( std::map< EcalLogicID, RunTTErrorsDat>* fillMap ) throw( std::runtime_error ) {

  if( !done_ ) {
    throw( std::runtime_error( "Input file not read" ) );
    return;
  }
  fillMap->clear();
  *fillMap = EcalErrorMaskFile::mapTTErrors_;
  return;
}

void EcalErrorMaskFile::fetchDataSet( std::map< EcalLogicID, RunPNErrorsDat>* fillMap ) throw( std::runtime_error ) {

  if( !done_ ) {
    throw( std::runtime_error( "Input file not read" ) );
    return;
  }
  fillMap->clear();
  *fillMap = EcalErrorMaskFile::mapPNErrors_;
  return;
}

void EcalErrorMaskFile::fetchDataSet( std::map< EcalLogicID, RunMemChErrorsDat>* fillMap ) throw( std::runtime_error ) {

  if( !done_ ) {
    throw( std::runtime_error( "Input file not read" ) );
    return;
  }
  fillMap->clear();
  *fillMap = EcalErrorMaskFile::mapMemChErrors_;
  return;
}

void EcalErrorMaskFile::fetchDataSet( std::map< EcalLogicID, RunMemTTErrorsDat>* fillMap ) throw( std::runtime_error ) {

  if( !done_ ) {
    throw( std::runtime_error( "Input file not read" ) );
    return;
  }
  fillMap->clear();
  *fillMap = EcalErrorMaskFile::mapMemTTErrors_;
  return;
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
  // From end of string, find last ' ' or '\t' (tab) and replace it with '\0'
  int i;
  for( i=strlen(line)-1; i>=0 && (line[i]==' '||line[i]=='\t'); i-- );
  if( line[i+1]  == ' ' || line[i+1] == '\t' ) line[i+1] = '\0';
}
