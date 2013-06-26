/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/07/20 02:58:22 $
 *  $Revision: 1.4 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/test/validate/DTMtimeValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTMtime.h"

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
DTMtimeValidateHandler::DTMtimeValidateHandler( const edm::ParameterSet& ps ):
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
DTMtimeValidateHandler::~DTMtimeValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTMtimeValidateHandler::getNewObjects() {
  int runNumber = firstRun;
  while ( runNumber <= lastRun ) addNewObject( runNumber++ );
  return;
}

void DTMtimeValidateHandler::addNewObject( int runNumber ) {

  DTMtime* mT = new DTMtime( dataVersion );

  std::stringstream run_fn;
  run_fn << "run" << runNumber << dataFileName;

  int status = 0;
  std::ofstream outFile( run_fn.str().c_str() );
  std::ofstream logFile( elogFileName.c_str(), std::ios_base::app );
  int whe;
  int sta;
  int sec;
  int qua;
  float mTime;
  float mTrms;
  float ckmt;
  float ckrms;

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
              mTime = random() * 10.0 / 0x0fffffff;
              mTrms = random() *  2.0 / 0x7fffffff;
              mTime += 370.0;
              status = mT->set( whe, sta, sec, qua,
                                mTime, mTrms, DTTimeUnits::counts );
              outFile << whe << " "
                      << sta << " "
                      << sec << " "
                      << qua << " "
                      << mTime << " "
                      << mTrms << std::endl;
              if ( status ) logFile << "ERROR while setting sl Mtime "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " , status = "
                                    << status << std::endl;
              status = mT->get( whe, sta, sec, qua,
                                ckmt, ckrms, DTTimeUnits::counts );
              if ( status ) logFile << "ERROR while checking sl Mtime "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " , status = "
                                    << status << std::endl;
              if ( ( fabs( ckmt  - mTime ) > 0.01   ) ||
                   ( fabs( ckrms - mTrms ) > 0.0001 ) )
                   logFile << "MISMATCH WHEN WRITING sl Mtime "
                           << whe << " "
                           << sta << " "
                           << sec << " "
                           << qua << " : "
                           << mTime << " " << mTrms << " -> "
                           << ckmt  << " " << ckrms << std::endl;
//            }
//          }
        }
      }
    }
  }

  cond::Time_t snc = runNumber;
  m_to_transfer.push_back( std::make_pair( mT, snc ) );

  return;

}


std::string DTMtimeValidateHandler::id() const {
  return dataVersion;
}


