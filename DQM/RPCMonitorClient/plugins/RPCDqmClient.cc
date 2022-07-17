// Package:    RPCDqmClient
// Original Author:  Anna Cimmino
#include "DQM/RPCMonitorClient/interface/RPCDqmClient.h"
#include "DQM/RPCMonitorClient/interface/RPCNameHelper.h"
#include "DQM/RPCMonitorClient/interface/RPCBookFolderStructure.h"
//include client headers
#include "DQM/RPCMonitorClient/interface/RPCDeadChannelTest.h"
#include "DQM/RPCMonitorClient/interface/RPCMultiplicityTest.h"
#include "DQM/RPCMonitorClient/interface/RPCClusterSizeTest.h"
#include "DQM/RPCMonitorClient/interface/RPCOccupancyTest.h"
#include "DQM/RPCMonitorClient/interface/RPCNoisyStripTest.h"
//Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
//Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <fmt/format.h>

RPCDqmClient::RPCDqmClient(const edm::ParameterSet& pset) {
  edm::LogVerbatim("rpcdqmclient") << "[RPCDqmClient]: Constructor";

  offlineDQM_ = pset.getUntrackedParameter<bool>("OfflineDQM", true);
  useRollInfo_ = pset.getUntrackedParameter<bool>("UseRollInfo", false);
  //check enabling
  enableDQMClients_ = pset.getUntrackedParameter<bool>("EnableRPCDqmClient", true);
  minimumEvents_ = pset.getUntrackedParameter<int>("MinimumRPCEvents", 10000);
  numberOfDisks_ = pset.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  numberOfRings_ = pset.getUntrackedParameter<int>("NumberOfEndcapRings", 2);

  std::string subsystemFolder = pset.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  std::string recHitTypeFolder = pset.getUntrackedParameter<std::string>("RecHitTypeFolder", "AllHits");
  std::string summaryFolder = pset.getUntrackedParameter<std::string>("SummaryFolder", "SummaryHistograms");

  prefixDir_ = subsystemFolder + "/" + recHitTypeFolder;
  globalFolder_ = subsystemFolder + "/" + recHitTypeFolder + "/" + summaryFolder;

  //get prescale factor
  prescaleGlobalFactor_ = pset.getUntrackedParameter<int>("DiagnosticGlobalPrescale", 5);

  //make default client list
  clientList_ = {{"RPCMultiplicityTest", "RPCDeadChannelTest", "RPCClusterSizeTest"}};
  clientList_ = pset.getUntrackedParameter<std::vector<std::string> >("RPCDqmClientList", clientList_);

  //get all the possible RPC DQM clients
  this->makeClientMap(pset);

  //clear counters
  lumiCounter_ = 0;

  rpcGeomToken_ = esConsumes<edm::Transition::EndLuminosityBlock>();
}

void RPCDqmClient::beginJob() {
  if (!enableDQMClients_) {
    return;
  };
  edm::LogVerbatim("rpcdqmclient") << "[RPCDqmClient]: Begin Job";

  //Do whatever the begin jobs of all client modules do
  for (auto& module : clientModules_) {
    module->beginJob(globalFolder_);
  }
}

void RPCDqmClient::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                         DQMStore::IGetter& igetter,
                                         edm::LuminosityBlock const& lumiSeg,
                                         edm::EventSetup const& c) {
  if (!enableDQMClients_) {
    return;
  }
  edm::LogVerbatim("rpcdqmclient") << "[RPCDqmClient]: End DQM LB";

  if (myDetIds_.empty()) {
    //Get RPCdetId...

    this->getRPCdetId(c);

    //...book summary histograms
    for (auto& module : clientModules_) {
      module->myBooker(ibooker);
    }
  }

  if (!offlineDQM_) {  //Do this only for the online

    if (lumiCounter_ == 0) {  //only for the first lumi section do this...
      // ...get chamber based histograms and pass them to the client modules
      this->getMonitorElements(igetter);
    }

    //Do not perform client oparations every lumi block
    ++lumiCounter_;
    if (lumiCounter_ % prescaleGlobalFactor_ != 0) {
      return;
    }

    //Check if there's enough statistics
    float rpcevents = minimumEvents_;
    if (RPCEvents_) {
      rpcevents = RPCEvents_->getBinContent(1);
    }
    if (rpcevents < minimumEvents_) {
      return;
    }

    edm::LogVerbatim("rpcdqmclient") << "[RPCDqmClient]: Client operations";
    for (auto& module : clientModules_) {
      module->clientOperation();
    }
  }  //end of online operations
}

void RPCDqmClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (!enableDQMClients_) {
    return;
  }

  edm::LogVerbatim("rpcdqmclient") << "[RPCDqmClient]: End DQM Job";

  if (offlineDQM_) {  // ...get chamber based histograms and pass them to the client modules
    this->getMonitorElements(igetter);
  }

  float rpcevents = minimumEvents_;
  if (RPCEvents_) {
    rpcevents = RPCEvents_->getBinContent(1);
  }
  if (rpcevents < minimumEvents_) {
    return;
  }

  edm::LogVerbatim("rpcdqmclient") << "[RPCDqmClient]: Client operations";
  for (auto& module : clientModules_) {
    module->clientOperation();
  }
}

void RPCDqmClient::getMonitorElements(DQMStore::IGetter& igetter) {
  std::vector<MonitorElement*> myMeVect;
  std::vector<RPCDetId> myDetIds;

  //loop on all geometry and get all histos
  for (auto& detId : myDetIds_) {
    //Get name
    const std::string rollName = RPCNameHelper::name(detId, useRollInfo_);
    const std::string folder = RPCBookFolderStructure::folderStructure(detId);

    //loop on clients
    for (unsigned int cl = 0, nCL = clientModules_.size(); cl < nCL; ++cl) {
      if (clientHisto_[cl] == "ClusterSize")
        continue;

      MonitorElement* myMe = igetter.get(fmt::format("{}/{}/{}_{}", prefixDir_, folder, clientHisto_[cl], rollName));
      if (!myMe)
        continue;

      myMeVect.push_back(myMe);
      myDetIds.push_back(detId);

    }  //end loop on clients
  }    //end loop on all geometry and get all histos

  //Clustersize
  std::vector<MonitorElement*> myMeVectCl;
  const std::array<std::string, 4> chNames = {{"CH01-CH09", "CH10-CH18", "CH19-CH27", "CH28-CH36"}};

  //Retrieve barrel clustersize
  for (int wheel = -2; wheel <= 2; wheel++) {
    for (int sector = 1; sector <= 12; sector++) {
      MonitorElement* myMeCl = igetter.get(fmt::format(
          "{}/Barrel/Wheel_{}/SummaryBySectors/ClusterSize_Wheel_{}_Sector_{}", prefixDir_, wheel, wheel, sector));
      myMeVectCl.push_back(myMeCl);
    }
  }
  //Retrieve endcaps clustersize
  for (int region = -1; region <= 1; region++) {
    if (region == 0)
      continue;

    std::string regionName = "Endcap-";
    if (region == 1)
      regionName = "Endcap+";

    for (int disk = 1; disk <= numberOfDisks_; disk++) {
      for (int ring = numberOfRings_; ring <= 3; ring++) {
        for (unsigned int ich = 0; ich < chNames.size(); ich++) {
          MonitorElement* myMeCl =
              igetter.get(fmt::format("{}/{}/Disk_{}/SummaryByRings/ClusterSize_Disk_{}_Ring_{}_{}",
                                      prefixDir_,
                                      regionName,
                                      (region * disk),
                                      (region * disk),
                                      ring,
                                      chNames[ich]));
          myMeVectCl.push_back(myMeCl);
        }
      }
    }
  }

  RPCEvents_ = igetter.get(prefixDir_ + "/RPCEvents");
  for (unsigned int cl = 0; cl < clientModules_.size(); ++cl) {
    if (clientHisto_[cl] == "ClusterSize")
      clientModules_[cl]->getMonitorElements(myMeVectCl, myDetIds, clientHisto_[cl]);
    else
      clientModules_[cl]->getMonitorElements(myMeVect, myDetIds, clientHisto_[cl]);
  }
}

void RPCDqmClient::getRPCdetId(const edm::EventSetup& eventSetup) {
  myDetIds_.clear();

  auto rpcGeo = eventSetup.getHandle(rpcGeomToken_);

  for (auto& det : rpcGeo->dets()) {
    const RPCChamber* ch = dynamic_cast<const RPCChamber*>(det);
    if (!ch)
      continue;

    //Loop on rolls in given chamber
    for (auto& r : ch->rolls()) {
      myDetIds_.push_back(r->id());
    }
  }
}

void RPCDqmClient::makeClientMap(const edm::ParameterSet& pset) {
  for (unsigned int i = 0; i < clientList_.size(); i++) {
    if (clientList_[i] == "RPCMultiplicityTest") {
      clientHisto_.push_back("Multiplicity");
      // clientTag_.push_back(rpcdqm::MULTIPLICITY);
      clientModules_.emplace_back(new RPCMultiplicityTest(pset));
    } else if (clientList_[i] == "RPCDeadChannelTest") {
      clientHisto_.push_back("Occupancy");
      clientModules_.emplace_back(new RPCDeadChannelTest(pset));
      // clientTag_.push_back(rpcdqm::OCCUPANCY);
    } else if (clientList_[i] == "RPCClusterSizeTest") {
      clientHisto_.push_back("ClusterSize");
      clientModules_.emplace_back(new RPCClusterSizeTest(pset));
      // clientTag_.push_back(rpcdqm::CLUSTERSIZE);
    } else if (clientList_[i] == "RPCOccupancyTest") {
      clientHisto_.push_back("Occupancy");
      clientModules_.emplace_back(new RPCOccupancyTest(pset));
      // clientTag_.push_back(rpcdqm::OCCUPANCY);
    } else if (clientList_[i] == "RPCNoisyStripTest") {
      clientHisto_.push_back("Occupancy");
      clientModules_.emplace_back(new RPCNoisyStripTest(pset));
      //clientTag_.push_back(rpcdqm::OCCUPANCY);
    }
  }

  return;
}
