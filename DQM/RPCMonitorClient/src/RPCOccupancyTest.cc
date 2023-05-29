/*  \author Anna Cimmino*/
#include <DQM/RPCMonitorClient/interface/RPCOccupancyTest.h>
#include "DQM/RPCMonitorClient/interface/RPCRollMapHisto.h"
#include "DQM/RPCMonitorClient/interface/utils.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

RPCOccupancyTest::RPCOccupancyTest(const edm::ParameterSet& ps) {
  edm::LogVerbatim("rpceventsummary") << "[RPCOccupancyTest]: Constructor";

  prescaleFactor_ = ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  numberOfRings_ = ps.getUntrackedParameter<int>("NumberOfEndcapRings", 2);
  useNormalization_ = ps.getUntrackedParameter<bool>("testMode", true);
  useRollInfo_ = ps.getUntrackedParameter<bool>("useRollInfo_", false);

  std::string subsystemFolder = ps.getUntrackedParameter<std::string>("RPCFolder", "RPC");
  std::string recHitTypeFolder = ps.getUntrackedParameter<std::string>("RecHitTypeFolder", "AllHits");

  prefixDir_ = subsystemFolder + "/" + recHitTypeFolder;
}

void RPCOccupancyTest::beginJob(std::string& workingFolder) {
  edm::LogVerbatim("rpceventsummary") << "[RPCOccupancyTest]: Begin job ";
  globalFolder_ = workingFolder;

  totalStrips_ = 0.;
  totalActive_ = 0.;

  Active_Dead = nullptr;
  Active_Fraction = nullptr;
}

void RPCOccupancyTest::getMonitorElements(std::vector<MonitorElement*>& meVector,
                                          std::vector<RPCDetId>& detIdVector,
                                          std::string& clientHistoName) {
  //Get NumberOfDigi ME for each roll
  for (unsigned int i = 0; i < meVector.size(); i++) {
    std::string meName = meVector[i]->getName();

    if (meName.find(clientHistoName) != std::string::npos) {
      myOccupancyMe_.push_back(meVector[i]);
      myDetIds_.push_back(detIdVector[i]);
    }
  }
}

void RPCOccupancyTest::clientOperation() {
  edm::LogVerbatim("rpceventsummary") << "[RPCOccupancyTest]: Client Operation";

  //Loop on MEs
  for (unsigned int i = 0; i < myOccupancyMe_.size(); i++) {
    this->fillGlobalME(myDetIds_[i], myOccupancyMe_[i]);
  }  //End loop on MEs

  //Active Channels
  if (Active_Fraction && totalStrips_ != 0.) {
    Active_Fraction->setBinContent(1, (totalActive_ / totalStrips_));
  }
  if (Active_Dead) {
    Active_Dead->setBinContent(1, totalActive_);
    Active_Dead->setBinContent(2, (totalStrips_ - totalActive_));
  }
}

void RPCOccupancyTest::myBooker(DQMStore::IBooker& ibooker) {
  ibooker.setCurrentFolder(globalFolder_);

  std::stringstream histoName;

  histoName.str("");
  histoName << "RPC_Active_Channel_Fractions";
  Active_Fraction = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(), 1, 0.5, 1.5);
  Active_Fraction->setBinLabel(1, "Active Fraction", 1);

  histoName.str("");
  histoName << "RPC_Active_Inactive_Strips";
  Active_Dead = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(), 2, 0.5, 2.5);
  Active_Dead->setBinLabel(1, "Active Strips", 1);
  Active_Dead->setBinLabel(2, "Inactive Strips", 1);

  for (int w = -2; w <= 2; w++) {  //loop on wheels

    histoName.str("");
    histoName << "AsymmetryLeftRight_Roll_vs_Sector_Wheel" << w;

    auto me = RPCRollMapHisto::bookBarrel(ibooker, w, histoName.str(), histoName.str(), useRollInfo_);
    AsyMeWheel[w + 2] = dynamic_cast<MonitorElement*>(me);
  }  //end Barrel

  for (int d = -numberOfDisks_; d <= numberOfDisks_; d++) {
    if (d == 0)
      continue;

    int offset = numberOfDisks_;
    if (d > 0)
      offset--;  //used to skip case equale to zero

    histoName.str("");
    histoName << "AsymmetryLeftRight_Ring_vs_Segment_Disk" << d;
    auto me = RPCRollMapHisto::bookEndcap(ibooker, d, histoName.str(), histoName.str(), useRollInfo_);
    AsyMeDisk[d + offset] = dynamic_cast<MonitorElement*>(me);
  }  //End loop on Endcap
}

void RPCOccupancyTest::fillGlobalME(RPCDetId& detId, MonitorElement* myMe) {
  if (!myMe)
    return;

  MonitorElement* AsyMe = nullptr;  //Left Right Asymetry

  if (detId.region() == 0) {
    AsyMe = AsyMeWheel[detId.ring() + 2];

  } else {
    if (-detId.station() + numberOfDisks_ >= 0) {
      if (detId.region() < 0) {
        AsyMe = AsyMeDisk[-detId.station() + numberOfDisks_];
      } else {
        AsyMe = AsyMeDisk[detId.station() + numberOfDisks_ - 1];
      }
    }
  }

  int xBin, yBin;
  if (detId.region() == 0) {  //Barrel
    xBin = detId.sector();
    rpcdqm::utils rollNumber;
    yBin = rollNumber.detId2RollNr(detId);
  } else {  //Endcap
    //get segment number
    RPCGeomServ RPCServ(detId);
    xBin = RPCServ.segment();
    (numberOfRings_ == 3 ? yBin = detId.ring() * 3 - detId.roll() + 1
                         : yBin = (detId.ring() - 1) * 3 - detId.roll() + 1);
  }

  int stripInRoll = myMe->getNbinsX();
  totalStrips_ += (float)stripInRoll;
  float FOccupancy = 0;
  float BOccupancy = 0;

  float totEnt = myMe->getEntries();
  for (int strip = 1; strip <= stripInRoll; strip++) {
    float stripEntries = myMe->getBinContent(strip);
    if (stripEntries > 0) {
      totalActive_++;
    }
    if (strip <= stripInRoll / 2) {
      FOccupancy += myMe->getBinContent(strip);
    } else {
      BOccupancy += myMe->getBinContent(strip);
    }
  }

  float asym = 0;
  if (totEnt != 0)
    asym = fabs((FOccupancy - BOccupancy) / totEnt);

  if (AsyMe)
    AsyMe->setBinContent(xBin, yBin, asym);
}
