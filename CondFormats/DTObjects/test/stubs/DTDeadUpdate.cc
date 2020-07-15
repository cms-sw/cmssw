
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/DTObjects/test/stubs/DTDeadUpdate.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"

namespace edmtest {

  DTDeadUpdate::DTDeadUpdate(edm::ParameterSet const& p) : dSum(nullptr) {}

  DTDeadUpdate::DTDeadUpdate(int i) : dSum(nullptr) {}

  DTDeadUpdate::~DTDeadUpdate() {}

  void DTDeadUpdate::analyze(const edm::Event& e, const edm::EventSetup& context) {
    if (dSum == nullptr)
      dSum = new DTDeadFlag("deadList");
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::ESHandle<DTDeadFlag> dList;
    context.get<DTDeadFlagRcd>().get(dList);
    std::cout << dList->version() << std::endl;
    std::cout << std::distance(dList->begin(), dList->end()) << " data in the container" << std::endl;
    DTDeadFlag::const_iterator iter = dList->begin();
    DTDeadFlag::const_iterator iend = dList->end();
    while (iter != iend) {
      const std::pair<DTDeadFlagId, DTDeadFlagData>& data = *iter++;
      const DTDeadFlagId& id = data.first;
      const DTDeadFlagData& st = data.second;
      std::cout << id.wheelId << " " << id.stationId << " " << id.sectorId << " " << id.slId << " " << id.layerId << " "
                << id.cellId << " -> " << st.dead_HV << " " << st.dead_TP << " " << st.dead_RO << " " << st.discCat
                << std::endl;
      if (st.dead_HV)
        dSum->setCellDead_HV(id.wheelId, id.stationId, id.sectorId, id.slId, id.layerId, id.cellId, true);
      if (st.dead_TP)
        dSum->setCellDead_TP(id.wheelId, id.stationId, id.sectorId, id.slId, id.layerId, id.cellId, true);
      if (st.dead_RO)
        dSum->setCellDead_RO(id.wheelId, id.stationId, id.sectorId, id.slId, id.layerId, id.cellId, true);
      if (st.discCat)
        dSum->setCellDiscCat(id.wheelId, id.stationId, id.sectorId, id.slId, id.layerId, id.cellId, true);
    }
  }
  void DTDeadUpdate::endJob() {
    std::cout << "DTDeadUpdate::endJob " << std::endl;
    edm::Service<cond::service::PoolDBOutputService> dbservice;
    if (!dbservice.isAvailable()) {
      std::cout << "db service unavailable" << std::endl;
      return;
    }

    fill_dead_HV("dead_HV_list.txt", dSum);
    fill_dead_TP("dead_TP_list.txt", dSum);
    fill_dead_RO("dead_RO_list.txt", dSum);
    fill_discCat("discCat_list.txt", dSum);

    if (dbservice->isNewTagRequest("DTDeadFlagRcd")) {
      dbservice->createNewIOV<DTDeadFlag>(dSum, dbservice->beginOfTime(), dbservice->endOfTime(), "DTDeadFlagRcd");
    } else {
      std::cout << "already present tag" << std::endl;
      int currentRun = 10;
      //      dbservice->appendTillTime<DTDeadFlag>(
      dbservice->appendSinceTime<DTDeadFlag>(dSum, currentRun, "DTDeadFlagRcd");
      //      dbservice->appendSinceTime<DTDeadFlag>(
      //                 dlist,dbservice->currentTime(),"DTDeadFlagRcd");
    }
  }

  void DTDeadUpdate::fill_dead_HV(const char* file, DTDeadFlag* deadList) {
    int status = 0;
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    std::ifstream ifile(file);
    while (ifile >> whe >> sta >> sec >> qua >> lay >> cel) {
      status = deadList->setCellDead_HV(whe, sta, sec, qua, lay, cel, true);
      std::cout << whe << " " << sta << " " << sec << " " << qua << " " << lay << " " << cel << "  -> ";
      std::cout << "insert status: " << status << std::endl;
    }
    return;
  }
  void DTDeadUpdate::fill_dead_TP(const char* file, DTDeadFlag* deadList) {
    int status = 0;
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    std::ifstream ifile(file);
    while (ifile >> whe >> sta >> sec >> qua >> lay >> cel) {
      status = deadList->setCellDead_TP(whe, sta, sec, qua, lay, cel, true);
      std::cout << whe << " " << sta << " " << sec << " " << qua << " " << lay << " " << cel << "  -> ";
      std::cout << "insert status: " << status << std::endl;
    }
    return;
  }
  void DTDeadUpdate::fill_dead_RO(const char* file, DTDeadFlag* deadList) {
    int status = 0;
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    std::ifstream ifile(file);
    while (ifile >> whe >> sta >> sec >> qua >> lay >> cel) {
      status = deadList->setCellDead_RO(whe, sta, sec, qua, lay, cel, true);
      std::cout << whe << " " << sta << " " << sec << " " << qua << " " << lay << " " << cel << "  -> ";
      std::cout << "insert status: " << status << std::endl;
    }
    return;
  }
  void DTDeadUpdate::fill_discCat(const char* file, DTDeadFlag* deadList) {
    int status = 0;
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    std::ifstream ifile(file);
    while (ifile >> whe >> sta >> sec >> qua >> lay >> cel) {
      status = deadList->setCellDiscCat(whe, sta, sec, qua, lay, cel, true);
      std::cout << whe << " " << sta << " " << sec << " " << qua << " " << lay << " " << cel << "  -> ";
      std::cout << "insert status: " << status << std::endl;
    }
    return;
  }

  DEFINE_FWK_MODULE(DTDeadUpdate);
}  // namespace edmtest
