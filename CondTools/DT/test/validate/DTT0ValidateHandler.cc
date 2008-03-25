/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/21 15:13:57 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/test/validate/DTT0ValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTT0.h"

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
DTT0ValidateHandler::DTT0ValidateHandler( const edm::ParameterSet& ps ):
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
DTT0ValidateHandler::~DTT0ValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTT0ValidateHandler::getNewObjects() {
  int runNumber = firstRun;
  while ( runNumber <= lastRun ) addNewObject( runNumber++ );
  return;
}

void DTT0ValidateHandler::addNewObject( int runNumber ) {

  DTT0* t0 = new DTT0( dataVersion );

  std::stringstream run_fn;
  run_fn << "run" << runNumber << dataFileName;

  int status = 0;
  std::ofstream outFile( run_fn.str().c_str() );
  std::ofstream logFile( elogFileName.c_str(), std::ios_base::app );
  int whe;
  int sta;
  int sec;
  int qua;
  int lay;
  int cel;
  float t0mean;
  float t0rms;
  float ckmean;
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
          lay = 5;
          while ( --lay ) {
            cel = 20;
            while ( --cel ) {
              t0mean = random() * 1.0 / 0x0fffffff;
              t0rms  = random() * 0.2 / 0x7fffffff;
              t0mean -= 4.0;
//              t0rms  /= 4.0;
              status = t0->set( whe, sta, sec, qua, lay, cel,
                                t0mean, t0rms );
              outFile << whe << " "
                      << sta << " "
                      << sec << " "
                      << qua << " "
                      << lay << " "
                      << cel << " "
                      << t0mean << " "
                      << t0rms  << std::endl;
              if ( status ) logFile << "ERROR while setting cell T0 "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " "
                                    << lay << " "
                                    << cel << " , status = "
                                    << status << std::endl;
              status = t0->get( whe, sta, sec, qua, lay, cel,
                                ckmean, ckrms );
              if ( status ) logFile << "ERROR while checking cell T0 "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " "
                                    << lay << " "
                                    << cel << " , status = "
                                    << status << std::endl;
              if ( ( ckmean != t0mean ) ||
                   ( ckrms  != t0rms  ) )
                   logFile << "MISMATCH WHEN WRITING cell T0 "
                           << whe << " "
                           << sta << " "
                           << sec << " "
                           << qua << " "
                           << lay << " "
                           << cel << " : "
                           << t0mean << " " << t0rms << " -> "
                           << ckmean << " " << ckrms << std::endl;
            }
          }
        }
      }
    }
  }

  cond::Time_t snc = runNumber;
  m_to_transfer.push_back( std::make_pair( t0, snc ) );

  return;

}


std::string DTT0ValidateHandler::id() const {
  return dataVersion;
}


