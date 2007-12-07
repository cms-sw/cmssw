/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/11/24 12:29:55 $
 *  $Revision: 1.1.2.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTStatusFlagHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <fstream>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTStatusFlagHandler::DTStatusFlagHandler( std::string name,
                                          std::string connect_string,
                                          const edm::Event& evt,
                                          const edm::EventSetup& est,
                                          const std::string& tag,
                                          const std::string& file ):
 popcon::PopConSourceHandler<DTStatusFlag>( name, connect_string,
                                            evt, est ),
 dataTag( tag ),
 fileName( file ) {
}

//--------------
// Destructor --
//--------------
DTStatusFlagHandler::~DTStatusFlagHandler() {
}

//--------------
// Operations --
//--------------
void DTStatusFlagHandler::getNewObjects() {

  int irun = event.id().run();
  int ievt = event.id().event();
  std::cout << "================ "
            << irun << " " << ievt << std::endl;

  std::map<std::string, popcon::PayloadIOV> mp = getOfflineInfo();
  std::map<std::string, popcon::PayloadIOV>::iterator iter = mp.begin();
  std::map<std::string, popcon::PayloadIOV>::iterator iend = mp.end();
  std::cout << "list of all tags: " << std::endl;
  while ( iter != iend ) {
    std::cout << "Tag: "                       << iter->first
              << " , last object valid since " << iter->second.last_since
              << " to "                        << iter->second.last_till
              << std::endl;
    iter++;
  }

  std::cout << "look for tag " << dataTag << std::endl;
  std::map<std::string, popcon::PayloadIOV>::iterator itag =
    mp.find( dataTag );

  DTStatusFlag* stFlag = new DTStatusFlag( dataTag );

  int status = 0;
  std::ifstream ifile( fileName.c_str() );
  int whe;
  int sta;
  int sec;
  int qua;
  int lay;
  int cel;
  int noiseFlag;
  int    feMask;
  int   tdcMask;
  int  trigMask;
  int  deadFlag;
  int  nohvFlag;
  while ( ifile >> whe
                >> sta
                >> sec
                >> qua
                >> lay
                >> cel
                >> noiseFlag
                >> feMask
                >> tdcMask
                >> trigMask
                >> deadFlag
                >> nohvFlag ) {
    status = stFlag->setCellStatus( whe, sta, sec, qua, lay, cel,
                                    noiseFlag != 0,
                                       feMask != 0,
                                      tdcMask != 0,
                                     trigMask != 0,
                                     deadFlag != 0,
                                     nohvFlag != 0 );
    std::cout << whe << " "
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
              <<  nohvFlag << "  -> ";                
    std::cout << "insert status: " << status << std::endl;
  }

  unsigned int runf = irun;
  unsigned int runl = 0xffffffff;
  popcon::IOVPair iop = { runf, runl };
  std::cout << "APPEND NEW OBJECT: "
            << runf << " " << runl << " " << stFlag << std::endl;
  m_to_transfer->push_back( std::make_pair( stFlag, iop ) );

  return;

}


