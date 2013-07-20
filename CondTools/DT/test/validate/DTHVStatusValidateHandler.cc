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
#include "CondTools/DT/test/validate/DTHVStatusValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTHVStatus.h"

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
DTHVStatusValidateHandler::DTHVStatusValidateHandler( const edm::ParameterSet& ps ):
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
DTHVStatusValidateHandler::~DTHVStatusValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTHVStatusValidateHandler::getNewObjects() {
  int runNumber = firstRun;
  while ( runNumber <= lastRun ) addNewObject( runNumber++ );
  return;
}

void DTHVStatusValidateHandler::addNewObject( int runNumber ) {

  DTHVStatus* hv = new DTHVStatus( dataVersion );

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
  int l_p;
  int fCell;
  int lCell;
  int flagA;
  int flagC;
  int flagS;
  int ckfCell;
  int cklCell;
  int ckflagA;
  int ckflagC;
  int ckflagS;

  whe = 3;
  while ( --whe >= -2 ) {
    sta = 5;
    while ( --sta ) {
      if ( sta == 4 ) sec = 15;
      else            sec = 13;
      while ( --sec ) {
	qua = 4;
        while ( --qua ) {
          lay = 5;
          while ( --lay ) {
            fCell = 1;
            lCell = 99;
            if ( ( sta == 4 ) &&
                 ( qua == 2 ) ) continue;
            if ( qua == 2 ) {
              lCell = 56;
            }
            else {
              if ( sta == 1 ) lCell = 48;
              if ( sta == 2 ) lCell = 60;
              if ( sta == 3 ) lCell = 72;
              if ( sta == 4 ) {
                if ( ( sec >=  1 ) && ( sec <=  3 ) ) lCell = 96;
                if ( ( sec ==  4 ) || ( sec == 13 ) ) lCell = 72;
                if ( ( sec >=  5 ) && ( sec <=  7 ) ) lCell = 96;
                if ( ( sec ==  8 ) || ( sec == 12 ) ) lCell = 92;
                if ( ( sec ==  9 ) || ( sec == 11 ) ) lCell = 48;
                if ( ( sec == 10 ) || ( sec == 14 ) ) lCell = 60;
              }
            }
            fCell = lCell / 2;
            l_p = 2;
            while ( l_p-- ) {
              if ( l_p == 0 ) {
                lCell = fCell - 1;
                fCell = ( lay == 4 ? 2 : 1 );
              }
              flagA = random() & 0x0000000f;
              flagC = random() & 0x0000000f;
              flagS = random() & 0x0000000f;
              if ( ( flagA == 3 ) || ( flagA >= 7 ) ) flagA = 0;
              if ( ( flagC == 3 ) || ( flagC >= 7 ) ) flagC = 0;
              if ( ( flagS == 3 ) || ( flagS >= 7 ) ) flagS = 0;
              if ( flagA ||
                   flagC ||
                   flagS ) status = hv->set( whe, sta, sec, qua, lay, l_p,
                                             fCell, lCell,
                                             flagA, flagC, flagS );
              else         status = 0;
              outFile << whe << " "
                      << sta << " "
                      << sec << " "
                      << qua << " "
                      << lay << " "
                      << l_p << " "
                      << fCell << " "
                      << lCell << " "
                      << flagA << " "
                      << flagC << " "
                      << flagS << std::endl;
              if ( status ) logFile << "ERROR while setting HV status"
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " "
                                    << lay << " "
                                    << l_p << " , status = "
                                    << status << std::endl;
              ckfCell = fCell;
              cklCell = lCell;
              ckflagA = ckflagC = ckflagS = 0;
              if ( flagA ||
                   flagC ||
                   flagS ) status = hv->get( whe, sta, sec, qua, lay, l_p,
                                             ckfCell, cklCell,
                                             ckflagA, ckflagC, ckflagS );
              else status = 0;
              if ( status ) logFile << "ERROR while checking HV status "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " "
                                    << lay << " "
                                    << l_p << " , status = "
                                    << status << std::endl;
              if ( ( ckfCell != fCell ) ||
                   ( cklCell != lCell ) ||
                   ( ckflagA != flagA ) ||
                   ( ckflagC != flagC ) ||
                   ( ckflagS != flagS ) )
                   logFile << "MISMATCH WHEN WRITING HV status "
                           << whe << " "
                           << sta << " "
                           << sec << " "
                           << qua << " "
                           << lay << " "
                           << l_p << " : "
                           << fCell << " "
                           << lCell << " "
                           << flagA << " "
                           << flagC << " "
                           << flagS << " -> "
                           << ckfCell << " "
                           << cklCell << " "
                           << ckflagA << " "
                           << ckflagC << " "
                           << ckflagS << std::endl;
            }
          }
        }
      }
    }
  }

  cond::Time_t snc = runNumber;
  m_to_transfer.push_back( std::make_pair( hv, snc ) );

  return;

}


std::string DTHVStatusValidateHandler::id() const {
  return dataVersion;
}


