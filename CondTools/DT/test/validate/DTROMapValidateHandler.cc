/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/25 16:19:57 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/test/validate/DTROMapValidateHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

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
DTROMapValidateHandler::DTROMapValidateHandler( const edm::ParameterSet& ps ):
 dataVersion(  ps.getParameter<std::string> ( "version" ) ),
 dataFileName( ps.getParameter<std::string> ( "outFile" ) ),
 elogFileName( ps.getParameter<std::string> ( "logFile" ) ) {
  std::ofstream logFile( elogFileName.c_str() );
}

//--------------
// Destructor --
//--------------
DTROMapValidateHandler::~DTROMapValidateHandler() {
}

//--------------
// Operations --
//--------------
void DTROMapValidateHandler::getNewObjects() {

  std::string dRosTag( dataVersion );
  std::string dRobTag( dataVersion );
  dRosTag += "_ROS";
  dRobTag += "_ROB";
  DTReadOutMapping* ro = new DTReadOutMapping( dRosTag, dRobTag );

//  std::stringstream run_fn;
//  run_fn << "run" << irun << dataFileName;

  int status = 0;
//  std::ofstream outFile( run_fn.str().c_str() );
  std::ofstream outFile( dataFileName.c_str() );
  std::ofstream logFile( elogFileName.c_str(), std::ios_base::app );
  int whe;
  int sta;
  int sec;
  int qua;
  int lay;
  int cel;
  int ddu;
  int ros;
  int rob;
  int tdc;
  int cha;
  int ckwhe;
  int cksta;
  int cksec;
  int ckqua;
  int cklay;
  int ckcel;
  int ckddu;
  int ckros;
  int ckrob;
  int cktdc;
  int ckcha;

  whe = 3;
  while ( --whe >= -2 ) {
    ddu = whe + 773;
    sta = 5;
    while ( --sta ) {
//      if ( sta == 4 ) sec = 15;
//      else            sec = 13;
      sec = 13;
      while ( --sec ) {
        ros = sec;
	qua = 4;
        while ( --qua ) {
          rob = ( ( sta - 1 ) * 6 ) + qua;
          if ( ( sta == 4 ) &&
               ( qua == 2 ) ) continue;
          lay = 5;
          while ( --lay ) {
            cel = 20;
            while ( --cel ) {
              cha = ( ( cel - 1 ) * 4 ) + lay;
              tdc = 0;
              while ( cha > 31 ) {
                cha -= 32;
                tdc++;
              }
              status = ro->insertReadOutGeometryLink(
                           ddu, ros, rob, tdc, cha,
                           whe, sta, sec, qua, lay, cel );
              outFile << ddu << " "
                      << ros << " "
                      << rob << " "
                      << tdc << " "
                      << cha << " "
                      << whe << " "
                      << sta << " "
                      << sec << " "
                      << qua << " "
                      << lay << " "
                      << cel << std::endl;
              if ( status ) logFile << "ERROR while setting chan-cell map "
                                    << ddu << " "
                                    << ros << " "
                                    << rob << " "
                                    << tdc << " "
                                    << cha << " -> "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " "
                                    << lay << " "
                                    << cel << " , status = "
                                    << status << std::endl;
              status = ro->readOutToGeometry(
                           ddu, ros, rob, tdc, cha,
                           ckwhe, cksta, cksec, ckqua, cklay, ckcel );
              if ( status ) logFile << "ERROR while checking chan->cell map "
                                    << ddu << " "
                                    << ros << " "
                                    << rob << " "
                                    << tdc << " "
                                    << cha << " -> "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " "
                                    << lay << " "
                                    << cel << " , status = "
                                    << status << std::endl;
              if ( ( ckwhe != whe ) ||
                   ( cksta != sta ) ||
                   ( cksec != sec ) ||
                   ( ckqua != qua ) ||
                   ( cklay != lay ) ||
                   ( ckcel != cel ) )
                   logFile << "MISMATCH WHEN WRITING chan->cell "
                           << ddu << " "
                           << ros << " "
                           << rob << " "
                           << tdc << " "
                           << cha << " : "
                           << whe << " "
                           << sta << " "
                           << sec << " "
                           << qua << " "
                           << lay << " "
                           << cel << " -> "
                           << ckwhe << " "
                           << cksta << " "
                           << cksec << " "
                           << ckqua << " "
                           << cklay << " "
                           << ckcel << std::endl;
              status = ro->geometryToReadOut(
                           whe, sta, sec, qua, lay, cel,
                           ckddu, ckros, ckrob, cktdc, ckcha );
              if ( status ) logFile << "ERROR while checking cell->chan map "
                                    << whe << " "
                                    << sta << " "
                                    << sec << " "
                                    << qua << " "
                                    << lay << " "
                                    << cel << " -> "
                                    << ddu << " "
                                    << ros << " "
                                    << rob << " "
                                    << tdc << " "
                                    << cha << " , status = "
                                    << status << std::endl;
              if ( ( ckddu != ddu ) ||
                   ( ckros != ros ) ||
                   ( ckrob != rob ) ||
                   ( cktdc != tdc ) ||
                   ( ckcha != cha ) )
                   logFile << "MISMATCH WHEN WRITING cell->chan "
                           << whe << " "
                           << sta << " "
                           << sec << " "
                           << qua << " "
                           << lay << " "
                           << cel << " : "
                           << ddu << " "
                           << ros << " "
                           << rob << " "
                           << tdc << " "
                           << cha << " -> "
                           << ckddu << " "
                           << ckros << " "
                           << ckrob << " "
                           << cktdc << " "
                           << ckcha << std::endl;
            }
          }
        }
      }
    }
  }

  cond::Time_t snc = 1;
  m_to_transfer.push_back( std::make_pair( ro, snc ) );

  return;

}


std::string DTROMapValidateHandler::id() const {
  return dataVersion;
}


