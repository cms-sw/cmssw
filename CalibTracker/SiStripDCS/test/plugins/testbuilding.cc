#include "CalibTracker/SiStripDCS/test/plugins/testbuilding.h"
#include "CalibTracker/SiStripDCS/interface/SiStripCoralIface.h"
#include "CalibTracker/SiStripDCS/interface/SiStripPsuDetIdMap.h"
#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVBuilder.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace std;
using namespace sistrip;

testbuilding::testbuilding(const edm::ParameterSet&) {}

testbuilding::~testbuilding() {}

void testbuilding::beginRun( const edm::Run& run, const edm::EventSetup& setup ) {
  hvBuilder->BuildModuleHVObj();
  //  std::vector< std::pair<SiStripModuleHV*,cond::Time_t> > resultVector = hvBuilder->getSiStripModuleHV();
  std::vector< std::pair<SiStripDetVOff*,cond::Time_t> > resultVector = hvBuilder->getModulesVOff();
  //  std::vector< std::vector<uint32_t> > stats = hvBuilder->getPayloadStats("HV");
  std::vector< std::vector<uint32_t> > stats = hvBuilder->getPayloadStats();
  
  std::cout << "Size of resultHV = " << resultVector.size() << std::endl;
  std::cout << "Size of stats    = " << stats.size() << std::endl << std::endl;
  for (unsigned int i = 0; i < resultVector.size(); i++) {
    std::cout << "Time is " << resultVector[i].second << std::endl;
    std::cout << "Number of bad det ids = " << stats[i][0] << std::endl;
    std::cout << "Number added          = " << stats[i][1] << std::endl;
    std::cout << "Number removed        = " << stats[i][2] << std::endl;
    std::vector<uint32_t> retVec;
    (resultVector[i].first)->getDetIds(retVec);
    std::cout << "Number of detids = " << retVec.size() << std::endl;
    for (unsigned int j = 0; j < retVec.size(); j++) {
      std::cout << "id = " << retVec[j] << " LV = " << (resultVector[i].first)->IsModuleLVOff(retVec[j]) << " HV = " << (resultVector[i].first)->IsModuleHVOff(retVec[j]) << std::endl;
    }
  }
  
}

