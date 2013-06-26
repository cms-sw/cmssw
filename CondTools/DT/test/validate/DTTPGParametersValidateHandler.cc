/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/07/20 02:58:22 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/test/validate/DTTPGParametersValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"

//---------------
// C++ Headers --
//---------------
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTTPGParametersValidateHandler::DTTPGParametersValidateHandler( const edm::ParameterSet& ps ):
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
DTTPGParametersValidateHandler::~DTTPGParametersValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTTPGParametersValidateHandler::getNewObjects() {
  int runNumber = firstRun;
  while ( runNumber <= lastRun ) addNewObject( runNumber++ );
  return;
}

void DTTPGParametersValidateHandler::addNewObject( int runNumber ) {

  DTTPGParameters* tpg = new DTTPGParameters( dataVersion );

  std::stringstream run_fn;
  run_fn << "run" << runNumber << dataFileName;

  int status = 0;
  std::ofstream outFile( run_fn.str().c_str() );
  std::ofstream logFile( elogFileName.c_str(), std::ios_base::app );
  int whe;
  int sta;
  int sec;
  int   nClock;
  float tPhase;
  int   ckclock;
  float ckphase;

  whe = 3;
  while ( --whe >= -2 ) {
    sta = 5;
    while ( --sta ) {
      if ( sta == 4 ) sec = 15;
      else            sec = 13;
      while ( --sec ) {
        nClock = random()        & 0x000000ff;
        tPhase = random() * 16.0 / 0x7fffffff;
        status = tpg->set( whe, sta, sec,
                           nClock, tPhase, DTTimeUnits::counts );
        outFile << whe << " "
                << sta << " "
                << sec << " "
                << nClock << " "
                << tPhase << std::endl;
        if ( status ) logFile << "ERROR while setting cell TPGParameters "
                              << whe << " "
                              << sta << " "
                              << sec << " , status = "
                              << status << std::endl;
        status = tpg->get( whe, sta, sec,
                           ckclock, ckphase, DTTimeUnits::counts );
        if ( status ) logFile << "ERROR while checking cell TPGParameters "
                              << whe << " "
                              << sta << " "
                              << sec << " , status = "
                              << status << std::endl;
        if ( ( fabs( ckclock - nClock ) > 0.0001 ) ||
             ( fabs( ckphase - tPhase ) > 0.0001 ) )
             logFile << "MISMATCH WHEN WRITING cell TPGParameters "
                     << whe << " "
                     << sta << " "
                     << sec << " : "
                     << nClock  << " " << tPhase  << " -> "
                     << ckclock << " " << ckphase << std::endl;
      }
    }
  }

  cond::Time_t snc = runNumber;
  m_to_transfer.push_back( std::make_pair( tpg, snc ) );

  return;

}


std::string DTTPGParametersValidateHandler::id() const {
  return dataVersion;
}


