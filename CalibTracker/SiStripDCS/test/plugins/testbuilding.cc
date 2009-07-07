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
  std::vector< std::pair<SiStripDetVOff*,cond::Time_t> > resultVector = hvBuilder->getModulesVOff();
  std::vector< std::vector<uint32_t> > stats = hvBuilder->getPayloadStats();
  
  std::cout << "Size of resultHV = " << resultVector.size() << std::endl;
  std::cout << "Size of stats    = " << stats.size() << std::endl << std::endl;
  for (unsigned int i = 0; i < resultVector.size(); i++) {
    std::cout << "Time is " << resultVector[i].second << " Index = " << i << std::endl;
    std::cout << "Number of bad det ids = " << stats[i][0] << std::endl;
    std::cout << "Number added          = " << stats[i][1] << std::endl;
    std::cout << "Number removed        = " << stats[i][2] << std::endl;
    std::vector<uint32_t> retVec;
    (resultVector[i].first)->getDetIds(retVec);
    std::vector<uint32_t> fullVec;
    (resultVector[i].first)->getVoff(fullVec);
    std::cout << "Number of detids = " << retVec.size() << std::endl;
    
    unsigned int LVbad = 0, HVbad = 0, allBad = 0;
    for (unsigned int j = 0; j < retVec.size(); j++) {
      if ((resultVector[i].first)->IsModuleLVOff(retVec[j]) && !((resultVector[i].first)->IsModuleHVOff(retVec[j])))  {LVbad++;}
      if (!((resultVector[i].first)->IsModuleLVOff(retVec[j])) && (resultVector[i].first)->IsModuleHVOff(retVec[j]))  {HVbad++;}
      if ((resultVector[i].first)->IsModuleLVOff(retVec[j]) && (resultVector[i].first)->IsModuleHVOff(retVec[j]))  {allBad++;}
    }

    std::cout << "Number LV bad         = " << LVbad << std::endl;
    std::cout << "Number HV bad         = " << HVbad << std::endl;
    std::cout << "Number both bad       = " << allBad << std::endl;

    if (i > 0) {
      SiStripDetVOff *currentV = resultVector[i].first;
      SiStripDetVOff *lastV = resultVector[i-1].first;
      std::vector<uint32_t> testDetID;
      currentV->getVoff(testDetID);
      std::vector<uint32_t> lastDetID;
      lastV->getVoff(lastDetID);
      if (testDetID == lastDetID) {std::cout << "vector for index = " << i << " matches the last one!" << std::endl;}
      if (*currentV == *lastV) {std::cout << "Value at end of pointer also match!" << std::endl;}
    }
  }
}

