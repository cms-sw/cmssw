
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CoralBase/TimeStamp.h"

#include "CondTools/DT/test/stubs/DTTimeUtility.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"
#include "CondFormats/DataRecord/interface/DTHVStatusRcd.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

namespace edmtest {

  DTTimeUtility::DTTimeUtility(edm::ParameterSet const& p)
      : year(p.getParameter<int>("year")),
        month(p.getParameter<int>("month")),
        day(p.getParameter<int>("day")),
        hour(p.getParameter<int>("hour")),
        min(p.getParameter<int>("minute")),
        sec(p.getParameter<int>("second")),
        condTime(p.getParameter<long long int>("condTime")),
        coralTime(p.getParameter<long long int>("coralTime")) {
    // parameters to setup
  }

  DTTimeUtility::DTTimeUtility(int i) {}

  DTTimeUtility::~DTTimeUtility() {}

  void DTTimeUtility::analyze(const edm::Event& e, const edm::EventSetup& context) {
    //    using namespace edm::eventsetup;
    long long int condConv = ((((condTime >> 32) & 0xFFFFFFFF) * 1000000000) + ((condTime & 0xFFFFFFFF) * 1000));
    coral::TimeStamp condStamp(condConv);
    std::cout << " condTime " << condTime << " -> coralTime " << condConv << " ( " << condStamp.year() << " , "
              << condStamp.month() << " , " << condStamp.day() << " h " << condStamp.hour() << ":" << condStamp.minute()
              << ":" << condStamp.second() << " ) " << std::endl;
    long long int coralConv = (((coralTime / 1000000000) << 32) + ((coralTime % 1000000000) / 1000));
    coral::TimeStamp convStamp(coralTime);
    std::cout << " coralTime " << coralTime << " -> condTime " << coralConv << " ( " << convStamp.year() << " , "
              << convStamp.month() << " , " << convStamp.day() << " h " << convStamp.hour() << ":" << convStamp.minute()
              << ":" << convStamp.second() << " ) " << std::endl;
    coral::TimeStamp coralStamp(year, month, day, hour, min, sec, 0);
    coralTime = coralStamp.total_nanoseconds();
    condTime = (((coralTime / 1000000000) << 32) + ((coralTime % 1000000000) / 1000));
    std::cout << coralStamp.year() << " , " << coralStamp.month() << " , " << coralStamp.day() << " h "
              << coralStamp.hour() << ":" << coralStamp.minute() << ":" << coralStamp.second() << " "
              << " -> coralTime " << coralTime << " -> condTime " << condTime << std::endl;
    return;
  }
  DEFINE_FWK_MODULE(DTTimeUtility);
}  // namespace edmtest
