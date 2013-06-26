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
#include "CondTools/DT/test/validate/DTRangeT0ValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTRangeT0.h"

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
DTRangeT0ValidateHandler::DTRangeT0ValidateHandler( const edm::ParameterSet& ps ):
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
DTRangeT0ValidateHandler::~DTRangeT0ValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTRangeT0ValidateHandler::getNewObjects() {
  int runNumber = firstRun;
  while ( runNumber <= lastRun ) addNewObject( runNumber++ );
  return;
}

void DTRangeT0ValidateHandler::addNewObject( int runNumber ) {

  DTRangeT0* tR = new DTRangeT0( dataVersion );

  std::stringstream run_fn;
  run_fn << "run" << runNumber << dataFileName;

  int status = 0;
  std::ofstream outFile( run_fn.str().c_str() );
  std::ofstream logFile( elogFileName.c_str(), std::ios_base::app );
  int whe;
  int sta;
  int sec;
  int qua;
  int t0min;
  int t0max;
  int ckt0min;
  int ckt0max;

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
              t0min =
              t0max = random() / 0x0000ffff;
              t0min -= 50.0;
              t0max += 50.0;
              status = tR->set( whe, sta, sec, qua,
                                t0min, t0max );
              outFile << whe << " "
                      << sta << " "
                      << sec << " "
                      << qua << " "
                      << t0min << " "
                      << t0max << std::endl;
              if ( status ) logFile << "ERROR while setting range T0"
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " , status = "
                                    << status << std::endl;
              status = tR->get( whe, sta, sec, qua,
                                ckt0min, ckt0max );
              if ( status ) logFile << "ERROR while checking range T0 "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " , status = "
                                    << status << std::endl;
              if ( ( fabs( ckt0min - t0min ) > 0.0001 ) ||
                   ( fabs( ckt0max - t0max ) > 0.0001 ) )
                   logFile << "MISMATCH WHEN WRITING range T0 "
                           << whe << " "
                           << sta << " "
                           << sec << " "
                           << qua << " : "
                           << t0min   << " "
                           << t0max   << " -> "
                           << ckt0min << " "
                           << ckt0max << std::endl;
//            }
//          }
        }
      }
    }
  }

  cond::Time_t snc = runNumber;
  m_to_transfer.push_back( std::make_pair( tR, snc ) );

  return;

}


std::string DTRangeT0ValidateHandler::id() const {
  return dataVersion;
}


