// system include files
#include <memory>

// user include files

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "CondTools/SiStrip/plugins/SiStripDetVOffFakeBuilder.h"

using namespace std;
using namespace cms;

SiStripDetVOffFakeBuilder::SiStripDetVOffFakeBuilder(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)), tkGeomToken_(esConsumes()) {}

SiStripDetVOffFakeBuilder::~SiStripDetVOffFakeBuilder() {}

void SiStripDetVOffFakeBuilder::initialize(const edm::EventSetup& iSetup) {
  const auto& tkGeom = iSetup.getData(tkGeomToken_);
  edm::LogInfo("SiStripDetVOffFakeBuilder") << " There are " << tkGeom.detUnits().size() << " detectors" << std::endl;

  for (const auto& it : tkGeom.detUnits()) {
    if (dynamic_cast<StripGeomDetUnit const*>(it) != nullptr) {
      uint32_t detid = (it->geographicalId()).rawId();
      const StripTopology& p = dynamic_cast<StripGeomDetUnit const*>(it)->specificTopology();
      unsigned short Nstrips = p.nstrips();
      if (Nstrips < 1 || Nstrips > 768) {
        edm::LogError("SiStripDetVOffFakeBuilder")
            << " Problem with Number of strips in detector.. " << p.nstrips() << " Exiting program" << endl;
        exit(1);
      }
      detids.push_back(detid);
      if (printdebug_)
        edm::LogInfo("SiStripDetVOffFakeBuilder") << "detid " << detid;
    }
  }
}

void SiStripDetVOffFakeBuilder::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
  initialize(iSetup);

  unsigned int run = evt.id().run();

  edm::LogInfo("SiStripDetVOffFakeBuilder")
      << "... creating dummy SiStripDetVOff Data for Run " << run << "\n " << std::endl;

  SiStripDetVOff* SiStripDetVOff_ = new SiStripDetVOff();

  // std::vector<uint32_t> TheDetIdHVVector;

  for (std::vector<uint32_t>::const_iterator it = detids.begin(); it != detids.end(); it++) {
    //Generate HV and LV for each channel, if at least one of the two is off fill the value
    int hv = rand() % 20;
    int lv = rand() % 20;
    if (hv <= 2) {
      edm::LogInfo("SiStripDetVOffFakeBuilder") << "detid: " << *it << " HV\t OFF" << std::endl;
      SiStripDetVOff_->put(*it, 1, -1);
      // TheDetIdHVVector.push_back(*it);
    }
    if (lv <= 2) {
      edm::LogInfo("SiStripDetVOffFakeBuilder") << "detid: " << *it << " LV\t OFF" << std::endl;
      SiStripDetVOff_->put(*it, -1, 1);
      // TheDetIdHVVector.push_back(*it);
    }
    if (lv <= 2 || hv <= 2)
      edm::LogInfo("SiStripDetVOffFakeBuilder") << "detid: " << *it << " V\t OFF" << std::endl;
  }

  // SiStripDetVOff_->put(TheDetIdHVVector);

  //End now write DetVOff data in DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;

  if (mydbservice.isAvailable()) {
    try {
      if (mydbservice->isNewTagRequest("SiStripDetVOffRcd")) {
        mydbservice->createNewIOV<SiStripDetVOff>(
            SiStripDetVOff_, mydbservice->beginOfTime(), mydbservice->endOfTime(), "SiStripDetVOffRcd");
      } else {
        mydbservice->appendSinceTime<SiStripDetVOff>(SiStripDetVOff_, mydbservice->currentTime(), "SiStripDetVOffRcd");
      }
    } catch (const cond::Exception& er) {
      edm::LogError("SiStripDetVOffFakeBuilder") << er.what() << std::endl;
    } catch (const std::exception& er) {
      edm::LogError("SiStripDetVOffFakeBuilder") << "caught std::exception " << er.what() << std::endl;
    }
  } else {
    edm::LogError("SiStripDetVOffFakeBuilder") << "Service is unavailable" << std::endl;
  }
}
