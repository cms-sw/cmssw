/*
 *  \author Anna Cimmino
 */
#include <DQM/RPCMonitorClient/interface/RPCMultiplicityTest.h>
#include <DQM/RPCMonitorClient/interface/RPCRollMapHisto.h>
#include "DQM/RPCMonitorClient/interface/utils.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

RPCMultiplicityTest::RPCMultiplicityTest(const edm::ParameterSet& ps) {
  edm::LogVerbatim("multiplicity") << "[RPCMultiplicityTest]: Constructor";
  useRollInfo_ = ps.getUntrackedParameter<bool>("UseRollInfo", false);
  prescaleFactor_ = ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);
  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  numberOfRings_ = ps.getUntrackedParameter<int>("NumberOfEndcapRings", 2);
  testMode_ = ps.getUntrackedParameter<bool>("testMode", false);
}

void RPCMultiplicityTest::beginJob(std::string& workingFolder) {
  edm::LogVerbatim("multiplicity") << "[RPCMultiplicityTest]: Begin job";
  globalFolder_ = workingFolder;
}

void RPCMultiplicityTest::myBooker(DQMStore::IBooker& ibooker) {
  ibooker.setCurrentFolder(globalFolder_);

  std::stringstream histoName;

  for (int i = -2; i <= 2; i++) {  //loop on wheels and disks

    histoName.str("");
    histoName << "NumberOfDigi_Mean_Roll_vs_Sector_Wheel" << i;

    auto me = RPCRollMapHisto::bookBarrel(ibooker, i, histoName.str(), histoName.str(), useRollInfo_);
    MULTWheel[i + 2] = dynamic_cast<MonitorElement*>(me);

    if (testMode_) {
      histoName.str("");
      histoName << "NumberOfDigi_Mean_Distribution_Wheel" << i;
      MULTDWheel[i + 2] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 0.5, 50.5);
    }

  }  //end wheels

  for (int d = -numberOfDisks_; d <= numberOfDisks_; d++) {
    if (d == 0)
      continue;

    int offset = numberOfDisks_;
    if (d > 0)
      offset--;  //used to skip case equale to zero

    histoName.str("");
    histoName << "NumberOfDigi_Mean_Ring_vs_Segment_Disk" << d;
    auto me = RPCRollMapHisto::bookEndcap(ibooker, d, histoName.str(), histoName.str(), useRollInfo_);
    MULTDisk[d + offset] = dynamic_cast<MonitorElement*>(me);

    if (testMode_) {
      histoName.str("");
      histoName << "NumberOfDigi_Mean_Distribution_Disk" << d;
      MULTDDisk[d + offset] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 0.5, 50.5);
    }
  }  //end loop on wheels and disks
}

void RPCMultiplicityTest::getMonitorElements(std::vector<MonitorElement*>& meVector,
                                             std::vector<RPCDetId>& detIdVector,
                                             std::string& clientHistoName) {
  //Get NumberOfDigi ME for each roll
  for (unsigned int i = 0; i < meVector.size(); i++) {
    std::string meName = meVector[i]->getName();

    if (meName.find(clientHistoName) != std::string::npos) {
      myNumDigiMe_.push_back(meVector[i]);
      myDetIds_.push_back(detIdVector[i]);
    }
  }
}

void RPCMultiplicityTest::clientOperation() {
  edm::LogVerbatim("multiplicity") << "[RPCMultiplicityTest]: Client Operation";

  //Loop on MEs
  for (unsigned int i = 0; i < myNumDigiMe_.size(); i++) {
    this->fillGlobalME(myDetIds_[i], myNumDigiMe_[i]);
  }  //End loop on MEs
}

void RPCMultiplicityTest::fillGlobalME(RPCDetId& detId, MonitorElement* myMe) {
  MonitorElement* MULT = nullptr;
  MonitorElement* MULTD = nullptr;

  if (detId.region() == 0) {
    MULT = MULTWheel[detId.ring() + 2];
    if (testMode_) {
      MULTD = MULTDWheel[detId.ring() + 2];
    }
  } else {
    if (-detId.station() + numberOfDisks_ >= 0) {
      if (detId.region() < 0) {
        MULT = MULTDisk[-detId.station() + numberOfDisks_];
        if (testMode_) {
          MULTD = MULTDDisk[-detId.station() + numberOfDisks_];
        }
      } else {
        MULT = MULTDisk[detId.station() + numberOfDisks_ - 1];
        if (testMode_) {
          MULTD = MULTDDisk[detId.station() + numberOfDisks_ - 1];
        }
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

  float mean = myMe->getMean();

  if (MULT) {
    MULT->setBinContent(xBin, yBin, mean);
  }
  if (testMode_ && MULTD) {
    MULTD->Fill(mean);
  }
}
