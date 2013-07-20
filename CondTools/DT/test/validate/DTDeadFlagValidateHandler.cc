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
#include "CondTools/DT/test/validate/DTDeadFlagValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"

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
DTDeadFlagValidateHandler::DTDeadFlagValidateHandler( const edm::ParameterSet& ps ):
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
DTDeadFlagValidateHandler::~DTDeadFlagValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTDeadFlagValidateHandler::getNewObjects() {
  int runNumber = firstRun;
  while ( runNumber <= lastRun ) addNewObject( runNumber++ );
  return;
}

void DTDeadFlagValidateHandler::addNewObject( int runNumber ) {

  DTDeadFlag* df = new DTDeadFlag( dataVersion );

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

  bool dead_HV;
  bool dead_TP;
  bool dead_RO;
  bool discCat;
  bool ckdead_HV;
  bool ckdead_TP;
  bool ckdead_RO;
  bool ckdiscCat;

  int sampleLimit = 100;

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
            cel = 48;
            while ( --cel ) {
              dead_HV = ( ( random() & 0x0000ffff ) < 500 );
              dead_TP = ( ( random() & 0x0000ffff ) < 500 );
              dead_RO = ( ( random() & 0x0000ffff ) < 500 );
              discCat = ( ( random() & 0x0000ffff ) < 500 );
              ckdead_HV = ckdead_TP = ckdead_RO = ckdiscCat = false;
              if ( dead_HV ||
                   dead_TP ||
                   dead_RO ||
                   discCat ) {
                if ( sampleLimit-- ) {
                  status = df->set( whe, sta, sec, qua, lay, cel,
                                    dead_HV, dead_TP, dead_RO, discCat );
                }
                else {
                  status = df->setCellDead_HV( whe, sta, sec, qua, lay, cel,
                                               dead_HV );
                  status = df->setCellDead_TP( whe, sta, sec, qua, lay, cel,
                                               dead_TP );
                  status = df->setCellDead_RO( whe, sta, sec, qua, lay, cel,
                                               dead_RO );
                  status = df->setCellDiscCat( whe, sta, sec, qua, lay, cel,
                                               discCat );
                }
              }
              else {
                status = 0;
              }

              outFile << whe << " "
                      << sta << " "
                      << sec << " "
                      << qua << " "
                      << lay << " "
                      << cel << " "
                      << dead_HV << " "
                      << dead_TP << " "
                      << dead_RO << " "
                      << discCat << std::endl;
              if ( status ) logFile << "ERROR while setting cell flags "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " "
                                    << lay << " "
                                    << cel << " , status = "
                                    << status << std::endl;
              if ( dead_HV ||
                   dead_TP ||
                   dead_RO ||
                   discCat )
                   status = df->get( whe, sta, sec, qua, lay, cel,
                                    ckdead_HV, ckdead_TP,
                                    ckdead_RO, ckdiscCat );
              else status = 0;
              if ( status ) logFile << "ERROR while checking cell flags "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " "
                                    << lay << " "
                                    << cel << " , status = "
                                    << status << std::endl;
              if ( ( ckdead_HV != dead_HV ) ||
                   ( ckdead_TP != dead_TP ) ||
                   ( ckdead_RO != dead_RO ) ||
                   ( ckdiscCat != discCat ) )
                   logFile << "MISMATCH WHEN WRITING cell flags "
                           << whe << " "
                           << sta << " "
                           << sec << " "
                           << qua << " "
                           << lay << " "
                           << cel << " : "
                           <<   dead_HV << " " <<   dead_TP << " "
                           <<   dead_RO << " " <<   discCat << " -> "
                           << ckdead_HV << " " << ckdead_TP << " "
                           << ckdead_RO << " " << ckdiscCat << std::endl;
//              }

            }
          }
        }
      }
    }
  }

  cond::Time_t snc = runNumber;
  m_to_transfer.push_back( std::make_pair( df, snc ) );

  return;

}


std::string DTDeadFlagValidateHandler::id() const {
  return dataVersion;
}


