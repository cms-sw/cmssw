/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/07/20 02:58:22 $
 *  $Revision: 1.5 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/test/validate/DTTtrigValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTTtrig.h"

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
DTTtrigValidateHandler::DTTtrigValidateHandler( const edm::ParameterSet& ps ):
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
DTTtrigValidateHandler::~DTTtrigValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTTtrigValidateHandler::getNewObjects() {
  int runNumber = firstRun;
  while ( runNumber <= lastRun ) addNewObject( runNumber++ );
  return;
}

void DTTtrigValidateHandler::addNewObject( int runNumber ) {

  DTTtrig* tT = new DTTtrig( dataVersion );

  std::stringstream run_fn;
  run_fn << "run" << runNumber << dataFileName;

  int status = 0;
  std::ofstream outFile( run_fn.str().c_str() );
  std::ofstream logFile( elogFileName.c_str(), std::ios_base::app );
  int whe;
  int sta;
  int sec;
  int qua;
  float tTrig;
  float tTrms;
  float kFact;
  float cktrig;
  float ckrms;
  float ckfact;

  whe = 3;
  while ( --whe >= -2 ) {
    sta = 5;
    while ( --sta ) {
      if ( sta == 4 ) sec = 15;
      else            sec = 13;
      while ( --sec ) {
	qua = 4;
        while ( --qua ) {
          if ( ( sta == 4 ) &&
               ( qua == 2 ) ) continue;
              tTrig = random() * 10.0 / 0x0fffffff;
              tTrms = random() *  2.0 / 0x7fffffff;
              tTrig += 4000.0;
              kFact = -0.7;
              status = tT->set( whe, sta, sec, qua,
                                tTrig, tTrms, kFact, DTTimeUnits::counts );
              outFile << whe << " "
                      << sta << " "
                      << sec << " "
                      << qua << " "
                      << tTrig << " "
                      << tTrms << " "
                      << kFact << std::endl;
              if ( status ) logFile << "ERROR while setting sl Ttrig "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " , status = "
                                    << status << std::endl;
              status = tT->get( whe, sta, sec, qua,
                                cktrig, ckrms, ckfact, DTTimeUnits::counts );
              if ( status ) logFile << "ERROR while checking sl Ttrig "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " , status = "
                                    << status << std::endl;
              if ( ( fabs( cktrig - tTrig ) > 0.0001 ) ||
                   ( fabs( ckrms  - tTrms ) > 0.0001 ) )
                   logFile << "MISMATCH WHEN WRITING sl Ttrig "
                           << whe << " "
                           << sta << " "
                           << sec << " "
                           << qua << " : "
                           << tTrig  << " "
                           << tTrms  << " "
                           << kFact  << " -> "
                           << cktrig << " "
                           << ckrms  << " "
                           << ckfact << std::endl;
//            }
//          }
        }
      }
    }
  }

  cond::Time_t snc = runNumber;
  m_to_transfer.push_back( std::make_pair( tT, snc ) );

  return;

}


std::string DTTtrigValidateHandler::id() const {
  return dataVersion;
}


