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

testbuilding::testbuilding(const edm::ParameterSet & pset) {
  hvBuilder = new SiStripModuleHVBuilder(pset);
}

testbuilding::~testbuilding() {
  delete hvBuilder;
}

void testbuilding::beginRun( const edm::Run& run, const edm::EventSetup& setup ) {
  hvBuilder->BuildModuleHVObj();
}

