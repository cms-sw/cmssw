/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/20 17:29:26 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/test/validate/DTCCBConfigValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <fstream>
#include <sstream>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTCCBConfigValidateHandler::DTCCBConfigValidateHandler( const edm::ParameterSet& ps ):
 firstRun(     ps.getParameter<unsigned int> ( "firstRun" ) ),
  lastRun(     ps.getParameter<unsigned int> (  "lastRun" ) ),
 dataVersion(  ps.getParameter<std::string> ( "version" ) ),
 dataFileName( ps.getParameter<std::string> ( "outFile" ) ),
 elogFileName( ps.getParameter<std::string> ( "logFile" ) ) {
  std::ofstream logFile( elogFileName.c_str() );
}

//--------------
// Destructor --
//--------------
DTCCBConfigValidateHandler::~DTCCBConfigValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTCCBConfigValidateHandler::getNewObjects() {
  int runNumber = firstRun;
  while ( runNumber <= lastRun ) addNewObject( runNumber++ );
  return;
}

void DTCCBConfigValidateHandler::addNewObject( int runNumber ) {

  DTCCBConfig* conf = new DTCCBConfig( dataVersion );

  std::stringstream run_fn;
  run_fn << "run" << runNumber << dataFileName;

  int status = 0;
  std::ofstream outFile( run_fn.str().c_str() );
  std::ofstream logFile( elogFileName.c_str(), std::ios_base::app );
  int whe;
  int sta;
  int sec;
  std::vector<DTConfigKey> fullConf;
  std::vector<int>         ccbConf;
  std::vector<int>         totConf;
  std::vector<int>         chkConf;
  int nbtype;
  int nbrick;
  int fkey;
  int bkey;

  nbtype = 7;
  fullConf.clear();
  fullConf.reserve( nbtype );
  outFile << --nbtype << std::endl;
  while ( nbtype ) {
    DTConfigKey confKey;
    fkey = random() & 0x00000fff;
    confKey.confType = nbtype;
    confKey.confKey  = fkey;
    fullConf.push_back( confKey );
    outFile << nbtype << " " << fkey << std::endl;
    nbtype--;
  }
  conf->setFullKey( fullConf );

  whe = 3;
  while ( --whe >= -2 ) {
    sta = 5;
    while ( --sta ) {
      if ( sta == 4 ) sec = 15;
      else            sec = 13;
      while ( --sec ) {
        nbrick = 12;
        ccbConf.clear();
        ccbConf.reserve( nbrick );
        totConf.clear();
        totConf.reserve( nbrick );
        outFile << whe << " "
                << sta << " "
                << sec << " "
                << --nbrick << std::endl;
        while ( nbrick > 6 ) {
          bkey = random() & 0x00000fff;
          ccbConf.push_back( bkey );
          totConf.push_back( bkey );
          outFile << bkey << std::endl;
          nbrick--;
        }
        status = conf->setConfigKey( whe, sta, sec, ccbConf );
        if ( status ) logFile << "ERROR while setting CCB configuration"
                              << whe << " "
                              << sta << " "
                              << sec << " , status = "
                              << status << std::endl;
        ccbConf.clear();
        while ( nbrick ) {
          bkey = random() & 0x00000fff;
          ccbConf.push_back( bkey );
          totConf.push_back( bkey );
          outFile << bkey << std::endl;
          nbrick--;
        }
        status = conf->appendConfigKey( whe, sta, sec, ccbConf );
        if ( status != -1 ) 
                      logFile << "ERROR while appending CCB configuration "
                              << whe << " "
                              << sta << " "
                              << sec << " , status = "
                              << status << std::endl;
        status = conf->configKey( whe, sta, sec, chkConf );
        if ( status ) logFile << "ERROR while checking CCB configuration "
                              << whe << " "
                              << sta << " "
                              << sec << " , status = "
                              << status << std::endl;
        if ( cfrDiff( totConf, chkConf ) )
             logFile << "MISMATCH WHEN WRITING CCB configuration "
                     << whe << " "
                     << sta << " "
                     << sec << std::endl;
           
      }
    }
  }

  cond::Time_t snc = runNumber;
  m_to_transfer.push_back( std::make_pair( conf, snc ) );

  return;

}


std::string DTCCBConfigValidateHandler::id() const {
  return dataVersion;
}


bool DTCCBConfigValidateHandler::cfrDiff( const std::vector<int>& l_conf,
                                          const std::vector<int>& r_conf ) {
  if ( l_conf.size() != r_conf.size() ) return true;
  std::vector<int>::const_iterator l_iter = l_conf.begin();
  std::vector<int>::const_iterator l_iend = l_conf.end();
  std::vector<int>::const_iterator r_iter = r_conf.begin();
  std::vector<int>::const_iterator r_iend = r_conf.end();
  while ( ( l_iter != l_iend ) && 
          ( r_iter != r_iend ) ) {
    if ( *l_iter++ != *r_iter++ ) return true;
  }
  return false;
}

