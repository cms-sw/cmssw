
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondFormats/DTObjects/test/stubs/DTMtimePrint.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"

namespace edmtest {

  DTMtimePrint::DTMtimePrint(edm::ParameterSet const& p) {
  }

  DTMtimePrint::DTMtimePrint(int i) {
  }

  DTMtimePrint::~DTMtimePrint() {
  }

  void DTMtimePrint::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<DTMtime> mTime;
    context.get<DTMtimeRcd>().get(mTime);
    std::cout << mTime->version() << std::endl;
    std::cout << std::distance( mTime->begin(), mTime->end() ) << " data in the container" << std::endl;
    DTMtime::const_iterator iter = mTime->begin();
    DTMtime::const_iterator iend = mTime->end();
    while ( iter != iend ) {
      const DTMtimeId&   mTimeId   = iter->first;
      const DTMtimeData& mTimeData = iter->second;
      float mTTime;
      float mTTrms;
      mTime->get( mTimeId.wheelId,
                  mTimeId.stationId,
                  mTimeId.sectorId,
                  mTimeId.slId,
                  mTimeId.layerId,
                  mTimeId.cellId,
//                  mTTime, mTTrms, DTVelocityUnits::cm_per_ns );
//                  mTTime, mTTrms, DTVelocityUnits::cm_per_count );
//                  mTTime, mTTrms, DTTimeUnits::ns );
                  mTTime, mTTrms, DTTimeUnits::counts );
      std::cout << mTimeId.wheelId   << " "
                << mTimeId.stationId << " "
                << mTimeId.sectorId  << " "
                << mTimeId.slId      << " "
                << mTimeId.layerId   << " "
                << mTimeId.cellId    << " -> "
                << mTimeData.mTime   << " "
                << mTimeData.mTrms   << " -> "
                << mTTime            << " "
                << mTTrms            << std::endl;
      iter++;
    }
  }
  DEFINE_FWK_MODULE(DTMtimePrint);
}
