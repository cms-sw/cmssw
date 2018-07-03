
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <math.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondTools/DT/test/validate/DTRangeT0ValidateDBRead.h"
#include "CondFormats/DTObjects/interface/DTRangeT0.h"
#include "CondFormats/DataRecord/interface/DTRangeT0Rcd.h"

DTRangeT0ValidateDBRead::DTRangeT0ValidateDBRead(edm::ParameterSet const& p):
 dataFileName( p.getParameter<std::string> ( "chkFile" ) ),
 elogFileName( p.getParameter<std::string> ( "logFile" ) ) {
}

DTRangeT0ValidateDBRead::DTRangeT0ValidateDBRead(int i) {
}

DTRangeT0ValidateDBRead::~DTRangeT0ValidateDBRead() {
}

void DTRangeT0ValidateDBRead::analyze(const edm::Event& e,
                        const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
  std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
  std::stringstream run_fn;
  run_fn << "run" << e.id().run() << dataFileName;
  std::ifstream chkFile( run_fn.str().c_str() );
  std::ofstream logFile( elogFileName.c_str(), std::ios_base::app );
  edm::ESHandle<DTRangeT0> tR;
  context.get<DTRangeT0Rcd>().get(tR);
  std::cout << tR->version() << std::endl;
  std::cout << std::distance( tR->begin(), tR->end() )
            << " data in the container" << std::endl;
  int status;
  int whe;
  int sta;
  int sec;
  int qua;
  int t0min;
  int t0max;
  int ckt0min;
  int ckt0max;

  DTRangeT0::const_iterator iter = tR->begin();
  DTRangeT0::const_iterator iend = tR->end();
  while ( iter != iend ) {
    const DTRangeT0Id&   tRId   = iter->first;
    const DTRangeT0Data& tRData = iter->second;
    status = tR->get( tRId.wheelId,
                      tRId.stationId,
                      tRId.sectorId,
                      tRId.slId,
                      t0min, t0max );
    if ( status ) logFile << "ERROR while getting range T0"
                          << tRId.wheelId   << " "
                          << tRId.stationId << " "
                          << tRId.sectorId  << " "
                          << tRId.slId      << " , status = "
                          << status << std::endl;
    if ( ( std::abs( tRData.t0min - t0min ) > 0.0001 ) ||
         ( std::abs( tRData.t0max - t0max ) > 0.0001 ) )
         logFile << "MISMATCH WHEN READING range T0"
                 << tRId.wheelId   << " "
                 << tRId.stationId << " "
                 << tRId.sectorId  << " "
                 << tRId.slId      << " : "
                 << tRData.t0min << " "
                 << tRData.t0max << " -> "
                 <<        t0min << " "
                 <<        t0max << std::endl;
    iter++;
  }

  while ( chkFile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> ckt0min
                  >> ckt0max ) {
    status = tR->get( whe,
                      sta,
                      sec,
                      qua,
                      t0min, t0max );
    if ( ( std::abs( ckt0min - t0min ) > 0.0001 ) ||
         ( std::abs( ckt0max - t0max ) > 0.0001 ) )
         logFile << "MISMATCH IN WRITING AND READING range T0 "
                 << whe << " "
                 << sta << " "
                 << sec << " "
                 << qua << " : "
                 << ckt0min << " "
                 << ckt0max << " -> "
                 << t0min   << " "
                 << t0max   << std::endl;
  }

}

void DTRangeT0ValidateDBRead::endJob() {
  std::ifstream logFile( elogFileName.c_str() );
  char* line = new char[1000];
  int errors = 0;
  std::cout << "Range T0 validation result:" << std::endl;
  while ( logFile.getline( line, 1000 ) ) {
    std::cout << line << std::endl;
    errors++;
  }
  if ( !errors ) {
    std::cout << " ********************************* " << std::endl;
    std::cout << " ***                           *** " << std::endl;
    std::cout << " ***      NO ERRORS FOUND      *** " << std::endl;
    std::cout << " ***                           *** " << std::endl;
    std::cout << " ********************************* " << std::endl;
  }
  return;
}

DEFINE_FWK_MODULE(DTRangeT0ValidateDBRead);
