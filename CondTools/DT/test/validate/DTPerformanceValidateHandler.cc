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
#include "CondTools/DT/test/validate/DTPerformanceValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTPerformance.h"

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
DTPerformanceValidateHandler::DTPerformanceValidateHandler( const edm::ParameterSet& ps ):
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
DTPerformanceValidateHandler::~DTPerformanceValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTPerformanceValidateHandler::getNewObjects() {
  int runNumber = firstRun;
  while ( runNumber <= lastRun ) addNewObject( runNumber++ );
  return;
}

void DTPerformanceValidateHandler::addNewObject( int runNumber ) {

  DTPerformance* mp = new DTPerformance( dataVersion );

  std::stringstream run_fn;
  run_fn << "run" << runNumber << dataFileName;

  int status = 0;
  std::ofstream outFile( run_fn.str().c_str() );
  std::ofstream logFile( elogFileName.c_str(), std::ios_base::app );
  int whe;
  int sta;
  int sec;
  int qua;
  float meanT0;
  float meanTtrig;
  float meanMtime;
  float meanNoise;
  float meanAfterPulse;
  float meanResolution;
  float meanEfficiency;
  float ckmeanT0;
  float ckmeanTtrig;
  float ckmeanMtime;
  float ckmeanNoise;
  float ckmeanAfterPulse;
  float ckmeanResolution;
  float ckmeanEfficiency;

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
              meanT0         = random() *   10.0 / 0x7fffffff;
              meanTtrig      = random() * 1000.0 / 0x0fffffff;
              meanMtime      = random() *   10.0 / 0x7fffffff;
              meanNoise      = random() *    1.0 / 0x0fffffff;
              meanAfterPulse = random() *    1.0 / 0x7fffffff;
              meanResolution = random() *   10.0 / 0x0fffffff;
              meanEfficiency = random() *   0.02 / 0x7fffffff;
              meanT0 -= 5.0;
              meanMtime += 470.0;
              meanEfficiency = 1.0 - meanEfficiency;
              status = mp->set( whe, sta, sec, qua,
                                meanT0, meanTtrig, meanMtime,
                                meanNoise, meanAfterPulse,
                                meanResolution, meanEfficiency,
                                DTTimeUnits::counts );
              outFile << whe << " "
                      << sta << " "
                      << sec << " "
                      << qua << " "
                      << meanT0 << " "
                      << meanTtrig << " "
                      << meanMtime << " "
                      << meanNoise << " "
                      << meanAfterPulse << " "
                      << meanResolution << " "
                      << meanEfficiency << std::endl;
              if ( status ) logFile << "ERROR while setting sl performance "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " , status = "
                                    << status << std::endl;
              status = mp->get( whe, sta, sec, qua,
                                ckmeanT0, ckmeanTtrig, ckmeanMtime,
                                ckmeanNoise, ckmeanAfterPulse,
                                ckmeanResolution, ckmeanEfficiency,
                                DTTimeUnits::counts );
              if ( status ) logFile << "ERROR while checking sl performance "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " , status = "
                                    << status << std::endl;
              if ( ( fabs( ckmeanT0         - meanT0         ) > 0.0001 ) ||
                   ( fabs( ckmeanTtrig      - meanTtrig      ) > 0.1    ) ||
                   ( fabs( ckmeanMtime      - meanMtime      ) > 0.01   ) ||
                   ( fabs( ckmeanNoise      - meanNoise      ) > 0.0001 ) ||
                   ( fabs( ckmeanAfterPulse - meanAfterPulse ) > 0.0001 ) ||
                   ( fabs( ckmeanResolution - meanResolution ) > 0.01   ) ||
                   ( fabs( ckmeanEfficiency - meanEfficiency ) > 0.0001 ) )
                   logFile << "MISMATCH WHEN WRITING sl performance "
                           << whe << " "
                           << sta << " "
                           << sec << " "
                           << qua << " : "
                           << meanT0          << " "
                           << meanTtrig       << " "
                           << meanMtime       << " "
                           << meanNoise       << " "
                           << meanAfterPulse  << " "
                           << meanResolution  << " "
                           << meanEfficiency  << " -> "
                           << ckmeanT0          << " "
                           << ckmeanTtrig       << " "
                           << ckmeanMtime       << " "
                           << ckmeanNoise       << " "
                           << ckmeanAfterPulse  << " "
                           << ckmeanResolution  << " "
                           << ckmeanEfficiency  << std::endl;
//            }
//          }
        }
      }
    }
  }

  cond::Time_t snc = runNumber;
  m_to_transfer.push_back( std::make_pair( mp, snc ) );

  return;

}


std::string DTPerformanceValidateHandler::id() const {
  return dataVersion;
}


