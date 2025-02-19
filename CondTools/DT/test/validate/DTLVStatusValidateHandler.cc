/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/20 18:20:09 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/test/validate/DTLVStatusValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTLVStatus.h"

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
DTLVStatusValidateHandler::DTLVStatusValidateHandler( const edm::ParameterSet& ps ):
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
DTLVStatusValidateHandler::~DTLVStatusValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTLVStatusValidateHandler::getNewObjects() {
  int runNumber = firstRun;
  while ( runNumber <= lastRun ) addNewObject( runNumber++ );
  return;
}

void DTLVStatusValidateHandler::addNewObject( int runNumber ) {

  DTLVStatus* lv = new DTLVStatus( dataVersion );

  std::stringstream run_fn;
  run_fn << "run" << runNumber << dataFileName;

  int status = 0;
  std::ofstream outFile( run_fn.str().c_str() );
  std::ofstream logFile( elogFileName.c_str(), std::ios_base::app );
  int whe;
  int sta;
  int sec;
  int flagCFE;
  int flagDFE;
  int flagCMC;
  int flagDMC;
  int ckflagCFE;
  int ckflagDFE;
  int ckflagCMC;
  int ckflagDMC;

  whe = 3;
  while ( --whe >= -2 ) {
    sta = 5;
    while ( --sta ) {
      if ( sta == 4 ) sec = 15;
      else            sec = 13;
      while ( --sec ) {
        flagCFE = random() & 0x0000000f;
        flagDFE = random() & 0x0000000f;
        flagCMC = random() & 0x0000000f;
        flagDMC = random() & 0x0000000f;
        if ( ( flagCFE == 3 ) || ( flagCFE >= 7 ) ) flagCFE = 0;
        if ( ( flagDFE == 3 ) || ( flagDFE >= 7 ) ) flagDFE = 0;
        if ( ( flagCMC == 3 ) || ( flagCMC >= 7 ) ) flagCMC = 0;
        if ( ( flagDMC == 3 ) || ( flagDMC >= 7 ) ) flagDMC = 0;
        if ( flagCFE ||
             flagDFE ||
             flagCMC ||
             flagDMC ) status = lv->set( whe, sta, sec,
                                         flagCFE, flagDFE, flagCMC, flagDMC );
        else         status = 0;
        outFile << whe << " "
                << sta << " "
                << sec << " "
                << flagCFE << " "
                << flagDFE << " "
                << flagCMC << " "
                << flagDMC << std::endl;
        if ( status ) logFile << "ERROR while setting LV status"
                              << whe << " "
                              << sta << " "
                              << sec << " , status = "
                              << status << std::endl;
        ckflagCFE = ckflagDFE = ckflagCMC = ckflagDMC = 0;
        if ( flagCFE ||
             flagDFE ||
             flagCMC ||
             flagDMC ) status = lv->get( whe, sta, sec,
                                         ckflagCFE, ckflagDFE,
                                         ckflagCMC, ckflagDMC );
        else status = 0;
        if ( status ) logFile << "ERROR while checking LV status "
                              << whe << " "
                              << sta << " "
                              << sec << " , status = "
                              << status << std::endl;
        if ( ( ckflagCFE != flagCFE ) ||
             ( ckflagDFE != flagDFE ) ||
             ( ckflagCMC != flagCMC ) ||
             ( ckflagDMC != flagDMC ) )
             logFile << "MISMATCH WHEN WRITING LV status "
                     << whe << " "
                     << sta << " "
                     << sec << " "
                     << flagCFE << " "
                     << flagDFE << " "
                     << flagCMC << " "
                     << flagDMC << " -> "
                     << ckflagCFE << " "
                     << ckflagDFE << " "
                     << ckflagCMC << " "
                     << ckflagDMC << std::endl;
      }
    }
  }

  cond::Time_t snc = runNumber;
  m_to_transfer.push_back( std::make_pair( lv, snc ) );

  return;

}


std::string DTLVStatusValidateHandler::id() const {
  return dataVersion;
}


