
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondTools/DT/test/stubs/DTHVDump.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"
#include "CondFormats/DataRecord/interface/DTHVStatusRcd.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

namespace edmtest {

  DTHVDump::DTHVDump(edm::ParameterSet const& p) {
// parameters to setup 
  }

  DTHVDump::DTHVDump(int i) {
  }

  DTHVDump::~DTHVDump() {
  }

  void DTHVDump::analyze( const edm::Event& e,
                          const edm::EventSetup& context ) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;

// get configuration for current run
    edm::ESHandle<DTHVStatus> hv;
    context.get<DTHVStatusRcd>().get(hv);
    std::cout << hv->version() << std::endl;
    std::cout << std::distance( hv->begin(), hv->end() )
              << " data in the container" << std::endl;
    edm::ValidityInterval iov(
         context.get<DTHVStatusRcd>().validityInterval() );
    unsigned int currValidityStart = iov.first().eventID().run();
    unsigned int currValidityEnd   = iov.last( ).eventID().run();
    std::cout << "valid since run " << currValidityStart
              << " to run "         << currValidityEnd << std::endl;

    DTHVStatus::const_iterator iter = hv->begin();
    DTHVStatus::const_iterator iend = hv->end();
    while ( iter != iend ) {
      const DTHVStatusId&   hvId   = iter->first;
      const DTHVStatusData& hvData = iter->second;
      std::cout << hvId.wheelId   << " "
                << hvId.stationId << " "
                << hvId.sectorId  << " "
                << hvId.slId      << " "
                << hvId.layerId   << " "
                << hvId.partId    << " -> "
                << hvData.fCell  << " "
                << hvData.lCell  << " "
                << hvData.flagA  << " "
                << hvData.flagC  << " "
                << hvData.flagS  << std::endl;
    }
    std::cout << "============" << std::endl;
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    int flagA;
    int flagC;
    int flagS;
    std::ifstream fcel( "cellList.txt" );
    while ( fcel >> whe >> sta >> sec >> qua >> lay >> cel ) {
      DTWireId id( whe, sta, sec, qua, lay, cel );
      hv->get( id, flagA, flagC, flagS );
      std::cout << whe   << " "
                << sta   << " "
                << sec   << " "
                << qua   << " "
                << lay   << " "
                << cel   << " -> "
                << flagA  << " "
                << flagC  << " "
                << flagS  << std::endl;
    }

    std::cout << "============" << std::endl;
    std::ifstream fcha( "chamList.txt" );
    while ( fcha >> whe >> sta >> sec ) {
      DTChamberId id( whe, sta, sec );
      std::cout << "chamber "
                << whe << " "
                << sta << " "
                << sec << " has "
                << hv->badChannelsNumber( id )
                << " bad cells and "
                << hv->offChannelsNumber( id )
                << " off cells" << std::endl;
    }
    std::cout << "============" << std::endl;
    std::cout << "total "
              << hv->badChannelsNumber()
              << " bad cells and "
              << hv->offChannelsNumber()
              << " off cells" << std::endl;

  }
  DEFINE_FWK_MODULE(DTHVDump);
}
