/**
 *
 * Analyzer that writes LUTs.
 *
 *\author L. Gray (4/13/06)
 *
 */

#include <fstream>
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "DataFormats/L1CSCTrackFinder/interface/CSCBitWidths.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"

#include "L1Trigger/CSCTrackFinder/test/analysis/CSCMakeSRLUT.h"

CSCMakeSRLUT::CSCMakeSRLUT(edm::ParameterSet const& conf) {
  writeLocalPhi = conf.getUntrackedParameter<bool>("WriteLocalPhi", true);
  station = conf.getUntrackedParameter<int>("Station", -1);
  sector = conf.getUntrackedParameter<int>("Sector", -1);
  endcap = conf.getUntrackedParameter<int>("Endcap", -1);
  isTMB07 = conf.getUntrackedParameter<bool>("isTMB07", false);
  writeGlobalPhi = conf.getUntrackedParameter<bool>("WriteGlobalPhi", true);
  writeGlobalEta = conf.getUntrackedParameter<bool>("WriteGlobalEta", true);
  binary = conf.getUntrackedParameter<bool>("BinaryOutput", true);
  LUTparam = conf.getParameter<edm::ParameterSet>("lutParam");
  geomToken_ = esConsumes();

  //init Sector Receiver LUTs
  for (int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    for (int se = CSCTriggerNumbering::minTriggerSectorId(); se <= CSCTriggerNumbering::maxTriggerSectorId(); ++se)
      for (int st = CSCDetId::minStationId(); st <= CSCDetId::maxStationId(); ++st) {
        if (st == 1)
          for (int ss = CSCTriggerNumbering::minTriggerSubSectorId();
               ss <= CSCTriggerNumbering::maxTriggerSubSectorId();
               ++ss) {
            mySR[e - 1][se - 1][ss - 1][st - 1] = new CSCSectorReceiverLUT(e, se, ss, st, LUTparam, isTMB07);
          }
        else {
          mySR[e - 1][se - 1][0][st - 1] = new CSCSectorReceiverLUT(e, se, 0, st, LUTparam, isTMB07);
          mySR[e - 1][se - 1][1][st - 1] = NULL;  // Save space.
        }
      }
}

CSCMakeSRLUT::~CSCMakeSRLUT() {
  for (int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    for (int se = CSCTriggerNumbering::minTriggerSectorId(); se <= CSCTriggerNumbering::maxTriggerSectorId(); ++se)
      for (int ss = CSCTriggerNumbering::minTriggerSubSectorId(); ss <= CSCTriggerNumbering::maxTriggerSubSectorId();
           ++ss)
        for (int st = CSCDetId::minStationId(); st <= CSCDetId::maxStationId(); ++st) {
          if (mySR[e - 1][se - 1][ss - 1][st - 1]) {
            delete mySR[e - 1][se - 1][ss - 1][st - 1];
            mySR[e - 1][se - 1][ss - 1][st - 1] = NULL;
          }
        }
}

void CSCMakeSRLUT::analyze(edm::Event const& e, edm::EventSetup const& iSetup) {
  edm::ESHandle<CSCGeometry> pDD = iSetup.getHandle(geomToken_);

  if (writeLocalPhi) {
    std::string filename = std::string("LocalPhiLUT") + ((binary) ? std::string(".bin") : std::string(".dat"));
    std::ofstream LocalPhiLUT(filename.c_str());
    for (int i = 0; i < 1 << CSCBitWidths::kLocalPhiAddressWidth; ++i) {
      unsigned short thedata;
      try {
        thedata = mySR[0][0][0][0]->localPhi(i).toint();
      } catch (...) {
        thedata = 0;
      }
      if (binary)
        LocalPhiLUT.write(reinterpret_cast<char*>(&thedata), sizeof(unsigned short));
      else
        LocalPhiLUT << std::dec << thedata << std::endl;
    }
  }

  if (writeGlobalPhi) {
    std::string MEprefix = "GlobalPhiME";
    std::string MBprefix = "GlobalPhiMB";
    std::ofstream GlobalPhiLUT;

    for (int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
      if (endcap == -1 || endcap == e)
        for (int se = CSCTriggerNumbering::minTriggerSectorId(); se <= CSCTriggerNumbering::maxTriggerSectorId(); ++se)
          if (sector == -1 || sector == se)
            for (int st = CSCDetId::minStationId(); st <= CSCDetId::maxStationId(); ++st)
              if (station == -1 || station == st)
                for (int ss = CSCTriggerNumbering::minTriggerSubSectorId();
                     ss <= CSCTriggerNumbering::maxTriggerSubSectorId();
                     ++ss) {
                  unsigned short thedata;
                  if (st == 1) {
                    std::string fname =
                        MEprefix + mySR[e - 1][se - 1][ss - 1][st - 1]->encodeFileIndex() + fileSuffix();
                    GlobalPhiLUT.open(fname.c_str());
                    for (int i = 0; i < 1 << CSCBitWidths::kGlobalPhiAddressWidth; ++i) {
                      try {
                        thedata = mySR[e - 1][se - 1][ss - 1][st - 1]->globalPhiME(i).toint();
                      } catch (...) {
                        thedata = 0;
                      }
                      if (binary)
                        GlobalPhiLUT.write(reinterpret_cast<char*>(&thedata), sizeof(unsigned short));
                      else
                        GlobalPhiLUT << std::dec << thedata << std::endl;
                    }
                    GlobalPhiLUT.close();

                    // Write MB global phi LUTs

                    fname = MBprefix + mySR[e - 1][se - 1][ss - 1][st - 1]->encodeFileIndex() + fileSuffix();
                    GlobalPhiLUT.open(fname.c_str());
                    for (int i = 0; i < 1 << CSCBitWidths::kGlobalPhiAddressWidth; ++i) {
                      try {
                        thedata = mySR[e - 1][se - 1][ss - 1][st - 1]->globalPhiMB(i).toint();
                      } catch (...) {
                        thedata = 0;
                      }
                      if (binary)
                        GlobalPhiLUT.write(reinterpret_cast<char*>(&thedata), sizeof(unsigned short));
                      else
                        GlobalPhiLUT << std::dec << thedata << std::endl;
                    }
                    GlobalPhiLUT.close();

                  } else {
                    if (ss == 1) {
                      std::string fname = MEprefix + mySR[e - 1][se - 1][0][st - 1]->encodeFileIndex() + fileSuffix();
                      GlobalPhiLUT.open(fname.c_str());
                      for (int i = 0; i < 1 << CSCBitWidths::kGlobalPhiAddressWidth; ++i) {
                        try {
                          thedata = mySR[e - 1][se - 1][0][st - 1]->globalPhiME(i).toint();
                        } catch (...) {
                          thedata = 0;
                        }
                        if (binary)
                          GlobalPhiLUT.write(reinterpret_cast<char*>(&thedata), sizeof(unsigned short));
                        else
                          GlobalPhiLUT << std::dec << thedata << std::endl;
                      }
                      GlobalPhiLUT.close();
                    }
                  }
                }
  }

  if (writeGlobalEta) {
    std::string prefix = "GlobalEtaME";
    std::ofstream GlobalEtaLUT;

    for (int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
      if (endcap == -1 || endcap == e)
        for (int se = CSCTriggerNumbering::minTriggerSectorId(); se <= CSCTriggerNumbering::maxTriggerSectorId(); ++se)
          if (sector == -1 || sector == se)
            for (int st = CSCDetId::minStationId(); st <= CSCDetId::maxStationId(); ++st)
              if (station == -1 || station == st)
                for (int ss = CSCTriggerNumbering::minTriggerSubSectorId();
                     ss <= CSCTriggerNumbering::maxTriggerSubSectorId();
                     ++ss) {
                  unsigned short thedata;
                  if (st == 1) {
                    std::string fname = prefix + mySR[e - 1][se - 1][ss - 1][st - 1]->encodeFileIndex() + fileSuffix();
                    GlobalEtaLUT.open(fname.c_str());
                    for (int i = 0; i < 1 << CSCBitWidths::kGlobalEtaAddressWidth; ++i) {
                      try {
                        thedata = mySR[e - 1][se - 1][ss - 1][st - 1]->globalEtaME(i).toint();
                      } catch (...) {
                        thedata = 0;
                      }
                      if (binary)
                        GlobalEtaLUT.write(reinterpret_cast<char*>(&thedata), sizeof(unsigned short));
                      else
                        GlobalEtaLUT << std::dec << thedata << std::endl;
                    }
                    GlobalEtaLUT.close();
                  } else {
                    if (ss == 1) {
                      std::string fname = prefix + mySR[e - 1][se - 1][0][st - 1]->encodeFileIndex() + fileSuffix();
                      GlobalEtaLUT.open(fname.c_str());
                      for (int i = 0; i < 1 << CSCBitWidths::kGlobalEtaAddressWidth; ++i) {
                        try {
                          thedata = mySR[e - 1][se - 1][0][st - 1]->globalEtaME(i).toint();
                        } catch (...) {
                          thedata = 0;
                        }
                        if (binary)
                          GlobalEtaLUT.write(reinterpret_cast<char*>(&thedata), sizeof(unsigned short));
                        else
                          GlobalEtaLUT << std::dec << thedata << std::endl;
                      }
                      GlobalEtaLUT.close();
                    }
                  }
                }
  }
}

std::string CSCMakeSRLUT::fileSuffix() const {
  std::string fileName = "";
  fileName += ((binary) ? ".bin" : ".dat");
  return fileName;
}
