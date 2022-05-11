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
  useRollInfo_ = ps.getUntrackedParameter<bool>("useRollInfo", true);

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
  for (unsigned int i = 0; i < meVector.size(); i++)
    myClusterMe_.push_back(meVector[i]);
}

void RPCClusterSizeTest::clientOperation() {
  edm::LogVerbatim("rpceventsummary") << "[RPCClusterSizeTest]:Client Operation";

  //check some statements and prescale Factor
  if (myClusterMe_.empty())
    return;

  MonitorElement* MEAN = nullptr;   // Mean ClusterSize, Roll vs Sector
  MonitorElement* MEAND = nullptr;  // Mean ClusterSize, Distribution

  //Loop on summary histograms
  for (unsigned int i = 0; i < myClusterMe_.size(); ++i) {
    MonitorElement* myMe = myClusterMe_[i];
    if (!myMe || myMe->getEntries() == 0)
      continue;
    std::string meName = myMe->getName();

    int xBin = 0, yBin = 0;
    float meanCLS = 0.;

    //Barrel
    for (int wheel = -2; wheel <= 2; wheel++) {
      for (int sector = 1; sector <= 12; sector++) {
        std::string tmpName = fmt::format("ClusterSize_Wheel_{}_Sector_{}", wheel, sector);

        if (tmpName == meName) {
          MEAN = MEANWheel[wheel + 2];
          if (testMode_) {
            MEAND = MEANDWheel[wheel + 2];
          }

          xBin = sector;
          for (int yBin_ = 1; yBin_ <= myMe->getNbinsY(); yBin_++) {
            int nbinsX = myMe->getNbinsX() - 1;  //Exclude overflow
            TH1F* h = new TH1F("h", "h", nbinsX, 0.5, 0.5 + nbinsX);
            for (int xBin_ = 1; xBin_ <= nbinsX; xBin_++) {
              int cl = myMe->getBinContent(xBin_, yBin_);
              h->SetBinContent(xBin_, cl);
            }
            yBin = yBin_;

            meanCLS = h->GetMean();

            if (MEAN)
              MEAN->setBinContent(xBin, yBin, meanCLS);

            if (testMode_) {
              if (MEAND)
                MEAND->Fill(meanCLS);
            }
          }
        }
      }
    }

    //Endcap
    const std::array<std::string, 4> chNames = {{"CH01-CH09", "CH10-CH18", "CH19-CH27", "CH28-CH36"}};

    for (int region = -1; region <= 1; region++) {
      if (region == 0)
        continue;

      for (int disk = 1; disk <= numberOfDisks_; disk++) {
        for (int ring = 2; ring < numberOfRings_ + 2; ring++) {
          for (unsigned int ich = 0; ich < chNames.size(); ich++) {
            std::string tmpName = fmt::format("ClusterSize_Disk_{}_Ring_{}_{}", (region * disk), ring, chNames[ich]);

            if (tmpName == meName) {
              xBin = (ich * 9);

              if (((disk * region) + numberOfDisks_) >= 0) {
                if (region < 0) {
                  MEAN = MEANDisk[(disk * region) + numberOfDisks_];
                  if (testMode_) {
                    MEAND = MEANDDisk[(disk * region) + numberOfDisks_];
                  }
                } else {
                  MEAN = MEANDisk[(disk * region) + numberOfDisks_ - 1];
                  if (testMode_) {
                    MEAND = MEANDDisk[(disk * region) + numberOfDisks_ - 1];
                  }
                }
              }

              for (int yBin_ = 1; yBin_ <= myMe->getNbinsY(); yBin_++) {
                int nbinsX = myMe->getNbinsX() - 1;  //Exclude overflow
                TH1F* h = new TH1F("h", "h", nbinsX, 0.5, 0.5 + nbinsX);
                for (int xBin_ = 1; xBin_ <= nbinsX; xBin_++) {
                  int cl = myMe->getBinContent(xBin_, yBin_);
                  h->SetBinContent(xBin_, cl);
                }

                yBin = yBin_ - (3 * ((yBin_ - 1) / 3)) + (3 * (ring - 2));
                if (yBin % 3 == 1)
                  xBin++;

                meanCLS = h->GetMean();

                if (MEAN)
                  MEAN->setBinContent(xBin, yBin, meanCLS);

                if (testMode_) {
                  if (MEAND)
                    MEAND->Fill(meanCLS);
                }
              }
            }  //name check
          }
        }
      }
    }
  }  //End loop on chambers
}

void RPCClusterSizeTest::resetMEArrays(void) {
  memset((void*)MEANWheel, 0, sizeof(MonitorElement*) * kWheels);
  memset((void*)MEANDWheel, 0, sizeof(MonitorElement*) * kWheels);

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
    histoName << "ClusterSizeMean_Roll_vs_Sector_Wheel" << w;  // Avarage ClusterSize (2D Roll vs Sector)
    auto me = RPCRollMapHisto::bookBarrel(ibooker, w, histoName.str(), histoName.str(), useRollInfo_);
    MEANWheel[w + 2] = dynamic_cast<MonitorElement*>(me);

    if (testMode_) {
      histoName.str("");
      histoName << "ClusterSizeMean_Distribution_Wheel" << w;  //  Avarage ClusterSize Distribution
      MEANDWheel[w + 2] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 1, 11);
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
    histoName << "ClusterSizeMean_Ring_vs_Segment_Disk" << d;  // Avarage ClusterSize (2D Roll vs Sector)
    auto me = RPCRollMapHisto::bookEndcap(ibooker, d, histoName.str(), histoName.str(), useRollInfo_);
    MEANDisk[d + offset] = dynamic_cast<MonitorElement*>(me);

    if (testMode_) {
      histoName.str("");
      histoName << "ClusterSizeMean_Distribution_Disk" << d;  //  Avarage ClusterSize Distribution
      MEANDDisk[d + offset] = ibooker.book1D(histoName.str().c_str(), histoName.str().c_str(), 100, 1, 11);
    }
  }
}
