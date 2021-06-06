#include "FWCore/Framework/interface/EventSetup.h"
#include "CalibTracker/SiStripESProducers/plugins/DBWriter/SiStripFedCablingManipulator.h"
#include <fstream>
#include <iostream>

SiStripFedCablingManipulator::SiStripFedCablingManipulator(const edm::ParameterSet& iConfig)
    : iConfig_(iConfig), fedCablingToken_(esConsumes<edm::Transition::EndRun>()) {
  edm::LogInfo("SiStripFedCablingManipulator") << "SiStripFedCablingManipulator constructor " << std::endl;
}

SiStripFedCablingManipulator::~SiStripFedCablingManipulator() {
  edm::LogInfo("SiStripFedCablingManipulator")
      << "SiStripFedCablingManipulator::~SiStripFedCablingManipulator()" << std::endl;
}

void SiStripFedCablingManipulator::endRun(const edm::Run& run, const edm::EventSetup& es) {
  const auto& esobj = es.getData(fedCablingToken_);

  auto obj = manipulate(esobj);

  cond::Time_t Time_ = 0;

  //And now write  data in DB
  edm::Service<cond::service::PoolDBOutputService> dbservice;
  if (dbservice.isAvailable()) {
    if (!obj) {
      edm::LogError("SiStripFedCablingManipulator") << "null pointer obj. nothing will be written " << std::endl;
      return;
    }

    std::string openIovAt = iConfig_.getUntrackedParameter<std::string>("OpenIovAt", "beginOfTime");
    if (openIovAt == "beginOfTime")
      Time_ = dbservice->beginOfTime();
    else if (openIovAt == "currentTime")
      dbservice->currentTime();
    else
      Time_ = iConfig_.getUntrackedParameter<uint32_t>("OpenIovAtTime", 1);

    //if first time tag is populated
    if (dbservice->isNewTagRequest("SiStripFedCablingRcd")) {
      edm::LogInfo("SiStripFedCablingManipulator") << "first request for storing objects with Record "
                                                   << "SiStripFedCablingRcd"
                                                   << " at time " << Time_ << std::endl;
      dbservice->createNewIOV<SiStripFedCabling>(obj.release(), Time_, dbservice->endOfTime(), "SiStripFedCablingRcd");
    } else {
      edm::LogInfo("SiStripFedCablingManipulator") << "appending a new object to existing tag "
                                                   << "SiStripFedCablingRcd"
                                                   << " in since mode " << std::endl;
      dbservice->appendSinceTime<SiStripFedCabling>(obj.release(), Time_, "SiStripFedCablingRcd");
    }
  } else {
    edm::LogError("SiStripFedCablingManipulator") << "Service is unavailable" << std::endl;
  }
}

std::unique_ptr<SiStripFedCabling> SiStripFedCablingManipulator::manipulate(const SiStripFedCabling& iobj) {
  std::string fp = iConfig_.getParameter<std::string>("file");
  std::unique_ptr<SiStripFedCabling> oobj;

  std::ifstream inputFile_;
  inputFile_.open(fp.c_str());

  std::map<uint32_t, std::pair<uint32_t, uint32_t> > dcuDetIdMap;
  uint32_t dcuid, Olddetid, Newdetid;

  // if(fp.c_str()==""){
  if (fp.empty()) {
    edm::LogInfo("SiStripFedCablingManipulator")
        << "::manipulate : since no file is specified, the copy of the input cabling will be applied" << std::endl;
    oobj = std::make_unique<SiStripFedCabling>(iobj);
  } else if (!inputFile_.is_open()) {
    edm::LogError("SiStripFedCablingManipulator") << "::manipulate - ERROR in opening file  " << fp << std::endl;
    throw cms::Exception("CorruptedData") << "::manipulate - ERROR in opening file  " << fp << std::endl;
  } else {
    for (;;) {
      inputFile_ >> dcuid >> Olddetid >> Newdetid;

      if (!(inputFile_.eof() || inputFile_.fail())) {
        if (dcuDetIdMap.find(dcuid) == dcuDetIdMap.end()) {
          edm::LogInfo("SiStripFedCablingManipulator") << dcuid << " " << Olddetid << " " << Newdetid << std::endl;

          dcuDetIdMap[dcuid] = std::pair<uint32_t, uint32_t>(Olddetid, Newdetid);
        } else {
          edm::LogError("SiStripFedCablingManipulator")
              << "::manipulate - ERROR duplicated dcuid " << dcuid << std::endl;
          throw cms::Exception("CorruptedData")
              << "SiStripFedCablingManipulator::manipulate - ERROR duplicated dcuid " << dcuid;
          break;
        }
      } else if (inputFile_.eof()) {
        edm::LogInfo("SiStripFedCablingManipulator") << "::manipulate - END of file reached" << std::endl;
        break;
      } else if (inputFile_.fail()) {
        edm::LogError("SiStripFedCablingManipulator") << "::manipulate - ERROR while reading file" << std::endl;
        break;
      }
    }
    inputFile_.close();
    std::map<uint32_t, std::pair<uint32_t, uint32_t> >::const_iterator it = dcuDetIdMap.begin();
    for (; it != dcuDetIdMap.end(); ++it)
      edm::LogInfo("SiStripFedCablingManipulator")
          << "::manipulate - Map " << it->first << " " << it->second.first << " " << it->second.second;

    std::vector<FedChannelConnection> conns;

    auto feds = iobj.fedIds();
    for (auto ifeds = feds.begin(); ifeds != feds.end(); ifeds++) {
      auto conns_per_fed = iobj.fedConnections(*ifeds);
      for (auto iconn = conns_per_fed.begin(); iconn != conns_per_fed.end(); ++iconn) {
        std::map<uint32_t, std::pair<uint32_t, uint32_t> >::const_iterator it = dcuDetIdMap.find(iconn->dcuId());
        if (it != dcuDetIdMap.end() && it->second.first == iconn->detId()) {
          edm::LogInfo("SiStripFedCablingManipulator")
              << "::manipulate - fedid " << *ifeds << " dcuid " << iconn->dcuId() << " oldDet " << iconn->detId()
              << " newDetID " << it->second.second;
          conns.push_back(FedChannelConnection(iconn->fecCrate(),
                                               iconn->fecSlot(),
                                               iconn->fecRing(),
                                               iconn->ccuAddr(),
                                               iconn->ccuChan(),
                                               iconn->i2cAddr(0),
                                               iconn->i2cAddr(1),
                                               iconn->dcuId(),
                                               it->second.second,  //<------ New detid
                                               iconn->nApvPairs(),
                                               iconn->fedId(),
                                               iconn->fedCh(),
                                               iconn->fiberLength(),
                                               iconn->dcu(),
                                               iconn->pll(),
                                               iconn->mux(),
                                               iconn->lld()));
        } else {
          conns.push_back(FedChannelConnection(iconn->fecCrate(),
                                               iconn->fecSlot(),
                                               iconn->fecRing(),
                                               iconn->ccuAddr(),
                                               iconn->ccuChan(),
                                               iconn->i2cAddr(0),
                                               iconn->i2cAddr(1),
                                               iconn->dcuId(),
                                               iconn->detId(),
                                               iconn->nApvPairs(),
                                               iconn->fedId(),
                                               iconn->fedCh(),
                                               iconn->fiberLength(),
                                               iconn->dcu(),
                                               iconn->pll(),
                                               iconn->mux(),
                                               iconn->lld()));
        }
      }
    }

    oobj = std::make_unique<SiStripFedCabling>(conns);
  }
  return oobj;
}
