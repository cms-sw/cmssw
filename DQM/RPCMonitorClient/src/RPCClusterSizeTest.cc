#include "DQM/RPCMonitorClient/interface/RPCClusterSizeTest.h"
#include <DQM/RPCMonitorClient/interface/RPCRollMapHisto.h>
#include "DQM/RPCMonitorClient/interface/utils.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

RPCClusterSizeTest::RPCClusterSizeTest(const edm::ParameterSet& ps) {
  edm::LogVerbatim("rpceventsummary") << "[RPCClusterSizeTest]: Constructor";

  prescaleFactor_ = ps.getUntrackedParameter<int>("DiagnosticPrescale", 1);

  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);
  numberOfRings_ = ps.getUntrackedParameter<int>("NumberOfEndcapRings", 2);
  testMode_ = ps.getUntrackedParameter<bool>("testMode", false);
  useRollInfo_ = ps.getUntrackedParameter<bool>("useRollInfo", false);

  resetMEArrays();
}

void RPCClusterSizeTest::beginJob(std::string& workingFolder) {
  edm::LogVerbatim("rpceventsummary") << "[RPCClusterSizeTest]: Begin job ";

  globalFolder_ = workingFolder;
}

void RPCClusterSizeTest::getMonitorElements(std::vector<MonitorElement*>& meVector,
                                            std::vector<RPCDetId>& detIdVector,
                                            std::string& clientHistoName) {
  //Get  ME for each roll
  for (unsigned int i = 0; i < meVector.size(); i++) {
    const std::string& meName = meVector[i]->getName();

    if (meName.find(clientHistoName) != std::string::npos) {
      myClusterMe_.push_back(meVector[i]);
      myDetIds_.push_back(detIdVector[i]);
    }
  }
}

void RPCClusterSizeTest::clientOperation() {
  edm::LogVerbatim("rpceventsummary") << "[RPCClusterSizeTest]:Client Operation";

  //check some statements and prescale Factor
  if (myClusterMe_.empty() || myDetIds_.empty())
    return;

  MonitorElement* CLS = nullptr;    // ClusterSize in 1 bin, Roll vs Sector
  MonitorElement* CLSD = nullptr;   // ClusterSize in 1 bin, Distribution
  MonitorElement* MEAN = nullptr;   // Mean ClusterSize, Roll vs Sector
  MonitorElement* MEAND = nullptr;  // Mean ClusterSize, Distribution

  std::stringstream meName;

  //Loop on chambers
  for (unsigned int i = 0; i < myClusterMe_.size(); ++i) {
    MonitorElement* myMe = myClusterMe_[i];
    if (!myMe || myMe->getEntries() == 0)
      continue;

    const RPCDetId& detId = myDetIds_[i];

    if (detId.region() == 0) {
      CLS = CLSWheel[detId.ring() + 2];
      MEAN = MEANWheel[detId.ring() + 2];
      if (testMode_) {
        CLSD = CLSDWheel[detId.ring() + 2];
        MEAND = MEANDWheel[detId.ring() + 2];
      }
    } else {
      if (((detId.station() * detId.region()) + numberOfDisks_) >= 0) {
        if (detId.region() < 0) {
          CLS = CLSDisk[(detId.station() * detId.region()) + numberOfDisks_];
          MEAN = MEANDisk[(detId.station() * detId.region()) + numberOfDisks_];
          if (testMode_) {
            CLSD = CLSDDisk[(detId.station() * detId.region()) + numberOfDisks_];
            MEAND = MEANDDisk[(detId.station() * detId.region()) + numberOfDisks_];
          }
        } else {
          CLS = CLSDisk[(detId.station() * detId.region()) + numberOfDisks_ - 1];
          MEAN = MEANDisk[(detId.station() * detId.region()) + numberOfDisks_ - 1];
          if (testMode_) {
            CLSD = CLSDDisk[(detId.station() * detId.region()) + numberOfDisks_ - 1];
            MEAND = MEANDDisk[(detId.station() * detId.region()) + numberOfDisks_ - 1];
          }
        }
      }
    }

    int xBin = 0, yBin = 0;

    if (detId.region() == 0) {  //Barrel

      rpcdqm::utils rollNumber;
      yBin = rollNumber.detId2RollNr(detId);
      xBin = detId.sector();
    } else {  //Endcap

      //get segment number
      RPCGeomServ RPCServ(detId);
      xBin = RPCServ.segment();
      (numberOfRings_ == 3 ? yBin = detId.ring() * 3 - detId.roll() + 1
                           : yBin = (detId.ring() - 1) * 3 - detId.roll() + 1);
    }

    // Normalization -> # of Entries in first Bin normalaized by total Entries
    const float NormCLS = myMe->getBinContent(1) / myMe->getEntries();
    const float meanCLS = myMe->getMean();

    if (CLS)
      CLS->setBinContent(xBin, yBin, NormCLS);
    if (MEAN)
      MEAN->setBinContent(xBin, yBin, meanCLS);

    if (testMode_) {
      if (MEAND)
        MEAND->Fill(meanCLS);
      if (CLSD)
        CLSD->Fill(NormCLS);
    }

  }  //End loop on chambers
}

void RPCClusterSizeTest::resetMEArrays(void) {
  memset((void*)CLSWheel, 0, sizeof(MonitorElement*) * kWheels);
  memset((void*)CLSDWheel, 0, sizeof(MonitorElement*) * kWheels);
  memset((void*)MEANWheel, 0, sizeof(MonitorElement*) * kWheels);
  memset((void*)MEANDWheel, 0, sizeof(MonitorElement*) * kWheels);

  memset((void*)CLSDisk, 0, sizeof(MonitorElement*) * kDisks);
  memset((void*)CLSDDisk, 0, sizeof(MonitorElement*) * kDisks);
  memset((void*)MEANDisk, 0, sizeof(MonitorElement*) * kDisks);
  memset((void*)MEANDDisk, 0, sizeof(MonitorElement*) * kDisks);
}

void RPCClusterSizeTest::myBooker(DQMStore::IBooker& ibooker) {
  resetMEArrays();

  ibooker.setCurrentFolder(globalFolder_);

  std::stringstream histoName;

  rpcdqm::utils rpcUtils;

  // Loop over wheels
  for (int w = -2; w <= 2; w++) {
    histoName.str("");
    histoName << "ClusterSizeIn1Bin_Roll_vs_Sector_Wheel"
              << w;  // ClusterSize in first bin norm. by Entries (2D Roll vs Sector)
    auto me = RPCRollMapHisto::bookBarrel(ibooker, w, histoName.str(), histoName.str(), useRollInfo_);
    CLSWheel[w + 2] = dynamic_cast<MonitorElement*>(me);

    histoName.str("");
    histoName << "ClusterSizeMean_Roll_vs_Sector_Wheel" << w;  // Avarage ClusterSize (2D Roll vs Sector)
    me = RPCRollMapHisto::bookBarrel(ibooker, w, histoName.str(), histoName.str(), useRollInfo_);
    MEANWheel[w + 2] = dynamic_cast<MonitorElement*>(me);

    if (testMode_) {
      histoName.str("");
      histoName << "ClusterSizeIn1Bin_Distribution_Wheel" << w;  //  ClusterSize in first bin, distribution
      CLSDWheel[w + 2] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(), 20, 0.0, 1.0);

      histoName.str("");
      histoName << "ClusterSizeMean_Distribution_Wheel" << w;  //  Avarage ClusterSize Distribution
      MEANDWheel[w + 2] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 0.5, 10.5);
    }
  }  //end loop on wheels

  for (int d = -numberOfDisks_; d <= numberOfDisks_; d++) {
    if (d == 0)
      continue;
    //Endcap
    int offset = numberOfDisks_;
    if (d > 0)
      offset--;

    histoName.str("");
    histoName << "ClusterSizeIn1Bin_Ring_vs_Segment_Disk"
              << d;  // ClusterSize in first bin norm. by Entries (2D Roll vs Sector)
    auto me = RPCRollMapHisto::bookEndcap(ibooker, d, histoName.str(), histoName.str(), useRollInfo_);
    CLSDisk[d + offset] = dynamic_cast<MonitorElement*>(me);

    if (testMode_) {
      histoName.str("");
      histoName << "ClusterSizeIn1Bin_Distribution_Disk" << d;  //  ClusterSize in first bin, distribution
      CLSDDisk[d + offset] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(), 20, 0.0, 1.0);

      histoName.str("");
      histoName << "ClusterSizeMean_Distribution_Disk" << d;  //  Avarage ClusterSize Distribution
      MEANDDisk[d + offset] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 0.5, 10.5);
    }

    histoName.str("");
    histoName << "ClusterSizeMean_Ring_vs_Segment_Disk" << d;  // Avarage ClusterSize (2D Roll vs Sector)
    me = RPCRollMapHisto::bookEndcap(ibooker, d, histoName.str(), histoName.str(), useRollInfo_);
    MEANDisk[d + offset] = dynamic_cast<MonitorElement*>(me);
  }
}
