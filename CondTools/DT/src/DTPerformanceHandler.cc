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
#include "CondTools/DT/interface/DTPerformanceHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTPerformance.h"

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
DTPerformanceHandler::DTPerformanceHandler( std::string name,
                                            std::string connect_string,
                                            const edm::Event& evt,
                                            const edm::EventSetup& est,
                                            const std::string& tag,
                                            const std::string& file ):
 popcon::PopConSourceHandler<DTPerformance>( name, connect_string,
                                             evt, est ),
 dataTag( tag ),
 fileName( file ) {
}

//--------------
// Destructor --
//--------------
DTPerformanceHandler::~DTPerformanceHandler() {
}

//--------------
// Operations --
//--------------
void DTPerformanceHandler::getNewObjects() {

  int irun = event.id().run();
  int ievt = event.id().event();
  std::cout << "================ "
            << irun << " " << ievt << std::endl;

//  edm::Service<cond::service::PoolDBOutputService> dbservice;

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

  DTPerformance* dtPerf = new DTPerformance( dataTag );

  int status = 0;
  std::ifstream ifile( fileName.c_str() );
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
  while ( ifile >> whe
                >> sta
                >> sec
                >> qua
                >> meanT0
                >> meanTtrig
                >> meanMtime
                >> meanNoise
                >> meanAfterPulse
                >> meanResolution
                >> meanEfficiency ) {
    status = dtPerf->set( whe, sta, sec, qua,
                          meanT0, meanTtrig, meanMtime, meanNoise,
                          meanAfterPulse, meanResolution, meanEfficiency );
    std::cout << whe << " "
              << sta << " "
              << sec << " "
              << qua << std::endl << " === "
              << meanT0 << " "
              << meanTtrig << " "
              << meanMtime << " "
              << meanNoise << " "
              << meanAfterPulse << " "
              << meanResolution << " "
              << meanEfficiency << std::endl << "  -> ";                
    std::cout << "insert status: " << status << std::endl;
  }

  unsigned int runf = irun;
  unsigned int runl = 0xffffffff;
  popcon::IOVPair iop = { runf, runl };
  std::cout << "APPEND NEW OBJECT: "
            << runf << " " << runl << " " << dtPerf << std::endl;
  m_to_transfer->push_back( std::make_pair( dtPerf, iop ) );

  return;

}


