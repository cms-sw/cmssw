/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/20 17:29:27 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/test/validate/DTStatusFlagValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

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
DTStatusFlagValidateHandler::DTStatusFlagValidateHandler( const edm::ParameterSet& ps ):
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
DTStatusFlagValidateHandler::~DTStatusFlagValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTStatusFlagValidateHandler::getNewObjects() {
  int runNumber = firstRun;
  while ( runNumber <= lastRun ) addNewObject( runNumber++ );
  return;
}

void DTStatusFlagValidateHandler::addNewObject( int runNumber ) {

  DTStatusFlag* sf = new DTStatusFlag( dataVersion );

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

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;
  bool cknoiseFlag;
  bool    ckfeMask;
  bool   cktdcMask;
  bool  cktrigMask;
  bool  ckdeadFlag;
  bool  cknohvFlag;

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
              noiseFlag = ( ( random() & 0x0000ffff ) < 500 );
                 feMask = ( ( random() & 0x0000ffff ) < 500 );
                tdcMask = ( ( random() & 0x0000ffff ) < 500 );
               trigMask = ( ( random() & 0x0000ffff ) < 500 );
               deadFlag = ( ( random() & 0x0000ffff ) < 500 );
               nohvFlag = ( ( random() & 0x0000ffff ) < 500 );
              cknoiseFlag =   ckfeMask =  cktdcMask =
               cktrigMask = ckdeadFlag = cknohvFlag = false;
              if ( noiseFlag ||
                      feMask ||
                     tdcMask ||
                    trigMask ||
                    deadFlag ||
                    nohvFlag ) {
                if ( sampleLimit-- ) {
                  status = sf->set( whe, sta, sec, qua, lay, cel,
                                    noiseFlag,   feMask,  tdcMask,
                                     trigMask, deadFlag, nohvFlag );
                }
                else {
                  status = sf->setCellNoise(    whe, sta, sec, qua, lay, cel,
                                                noiseFlag );
                  status = sf->setCellFEMask(   whe, sta, sec, qua, lay, cel,
                                                   feMask );
                  status = sf->setCellTDCMask(  whe, sta, sec, qua, lay, cel,
                                                  tdcMask );
                  status = sf->setCellTrigMask( whe, sta, sec, qua, lay, cel,
                                                 trigMask );
                  status = sf->setCellDead(     whe, sta, sec, qua, lay, cel,
                                                 deadFlag );
                  status = sf->setCellNoHV(     whe, sta, sec, qua, lay, cel,
                                                 nohvFlag );
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
                      << noiseFlag << " "
                      <<    feMask << " "
                      <<   tdcMask << " "
                      <<  trigMask << " "
                      <<  deadFlag << " "
                      <<  nohvFlag << std::endl;
              if ( status ) logFile << "ERROR while setting cell flags "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " "
                                    << lay << " "
                                    << cel << " , status = "
                                    << status << std::endl;
              if ( noiseFlag ||
                      feMask ||
                     tdcMask ||
                    trigMask ||
                    deadFlag ||
                    nohvFlag )
                   status = sf->get( whe, sta, sec, qua, lay, cel,
                                     cknoiseFlag,   ckfeMask,  cktdcMask,
                                      cktrigMask, ckdeadFlag, cknohvFlag );
              else status = 0;
              if ( status ) logFile << "ERROR while checking cell flags "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " "
                                    << lay << " "
                                    << cel << " , status = "
                                    << status << std::endl;
              if ( ( cknoiseFlag != noiseFlag ) ||
                   (    ckfeMask !=    feMask ) ||
                   (   cktdcMask !=   tdcMask ) ||
                   (  cktrigMask !=  trigMask ) ||
                   (  ckdeadFlag !=  deadFlag ) ||
                   (  cknohvFlag !=  nohvFlag ) )
                   logFile << "MISMATCH WHEN WRITING cell flags "
                           << whe << " "
                           << sta << " "
                           << sec << " "
                           << qua << " "
                           << lay << " "
                           << cel << " : "
                           <<   noiseFlag << " " <<     feMask << " "
                           <<     tdcMask << " " <<   trigMask << " "
                           <<    deadFlag << " " <<   nohvFlag << " -> "
                           << cknoiseFlag << " " <<   ckfeMask << " "
                           <<   cktdcMask << " " << cktrigMask << " "
                           <<  ckdeadFlag << " " << cknohvFlag << std::endl;
//              }

            }
          }
        }
      }
    }
  }

  cond::Time_t snc = runNumber;
  m_to_transfer.push_back( std::make_pair( sf, snc ) );

  return;

}


std::string DTStatusFlagValidateHandler::id() const {
  return dataVersion;
}


