#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/CSCDMBHeader.h"
#include "EventFilter/CSCRawToDigi/interface/cscPackerCompare.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2006.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2007.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2007_rev0x50c3.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2013.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2020_TMB.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2020_CCLUT.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2020_GEM.h"
#include "EventFilter/CSCRawToDigi/interface/CSCTMBHeader2020_Run2.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>
#include <cstring>  // memcpy

#ifdef LOCAL_UNPACK
bool CSCTMBHeader::debug = false;
#else
std::atomic<bool> CSCTMBHeader::debug{false};
#endif

CSCTMBHeader::CSCTMBHeader(int firmwareVersion, int firmwareRevision)
    : theHeaderFormat(), theFirmwareVersion(firmwareVersion) {
  if (firmwareVersion == 2020) {
    if ((firmwareRevision < 0x4000) && (firmwareRevision > 0x0)) { /* New (O)TMB firmware revision format */
      bool isGEM_fw = false;
      bool isCCLUT_HMT_fw = false;
      bool isOTMB_Run2_fw = false;
      bool isTMB_Run3_fw = false;
      bool isTMB_Run2_fw = false;
      bool isTMB_hybrid_fw = false;  /// Copper TMB hybrid fw Run2 CLCT + Run3 LCT/MPC + anode-only HMT (March 2022)
      bool isRun2_df = false;
      unsigned df_version = (firmwareRevision >> 9) & 0xF;  // 4-bits Data Format version
      unsigned major_ver = (firmwareRevision >> 5) & 0xF;   // 4-bits major version part
      // unsigned minor_ver = firmwareRevision & 0x1F;         // 5-bits minor version part
      switch (df_version) {
        case 0x4:
          isTMB_hybrid_fw = true;
          break;
        case 0x3:
          isGEM_fw = true;
          break;
        case 0x2:
          isCCLUT_HMT_fw = true;
          break;
        case 0x1:
          isOTMB_Run2_fw = true;
          break;
        case 0x0:
          if (major_ver == 1)
            isTMB_Run2_fw = true;
          else
            isTMB_Run3_fw = true;
          break;
        default:
          isGEM_fw = true;
      }
      if (major_ver == 1) {
        isRun2_df = true;
      }

      if (isGEM_fw) {
        if (isRun2_df) {
          theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_Run2(firmwareRevision));
        } else {
          theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_GEM());
        }
      } else if (isCCLUT_HMT_fw) {
        if (isRun2_df) {
          theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_Run2(firmwareRevision));
        } else {
          theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_CCLUT());
        }
      } else if (isOTMB_Run2_fw || isTMB_Run2_fw || isRun2_df) {
        theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_Run2(firmwareRevision));
      } else if (isTMB_Run3_fw) {
        theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_CCLUT());
      } else if (isTMB_hybrid_fw) {
        theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_TMB());
      }
    }
  } else if (firmwareVersion == 2013) {
    theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2013());
  } else if (firmwareVersion == 2006) {
    theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2006());
  } else if (firmwareVersion == 2007) {
    /* Checks for TMB2007 firmware revisions ranges to detect data format
       * rev.0x50c3 - first revision with changed format
       * rev.0x42D5 - oldest known from 06/21/2007
       * There is 4-bits year value rollover in revision number (0 in 2016)
       */
    if ((firmwareRevision >= 0x50c3) || (firmwareRevision < 0x42D5)) {
      // if (firmwareRevision >= 0x7a76) // First OTMB firmware revision with 2013 format
      /* Revisions > 0x6000 - OTMB firmwares, < 0x42D5 - new TMB revisions in 2016 */
      if ((firmwareRevision >= 0x6000) || (firmwareRevision < 0x42D5)) {
        bool isGEMfirmware = false;
        /* There are OTMB2013 firmware versions exist, which reports firmwareRevision code = 0x0 */
        if ((firmwareRevision < 0x4000) && (firmwareRevision > 0x0)) { /* New (O)TMB firmware revision format */
          if (((firmwareRevision >> 9) & 0x3) == 0x3)
            isGEMfirmware = true;
          if (isGEMfirmware) {
            theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_GEM());
          } else {
            theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2013());
          }
        }
      } else {
        theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2007_rev0x50c3());
      }
    } else {
      theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2007());
    }
  } else {
    edm::LogError("CSCTMBHeader|CSCRawToDigi") << "failed to determine TMB firmware version!!";
  }
}

//CSCTMBHeader::CSCTMBHeader(const CSCTMBStatusDigi & digi) {
//  CSCTMBHeader(digi.header());
//}

CSCTMBHeader::CSCTMBHeader(const unsigned short *buf) : theHeaderFormat() {
  ///first determine the format
  if (buf[0] == 0xDB0C) {
    theFirmwareVersion = 2007;
    theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2007(buf));
    /* Checks for TMB2007 firmware revisions ranges to detect data format
       * rev.0x50c3 - first revision with changed format
       * rev.0x42D5 - oldest known from 06/21/2007
       * There is 4-bits year value rollover in revision number (0 in 2016)
       */
    if ((theHeaderFormat->firmwareRevision() >= 0x50c3) || (theHeaderFormat->firmwareRevision() < 0x42D5)) {
      // if (theHeaderFormat->firmwareRevision() >= 0x7a76) // First OTMB firmware revision with 2013 format
      /* Revisions > 0x6000 - OTMB firmwares, < 0x42D5 - new TMB revisions in 2016 */
      if ((theHeaderFormat->firmwareRevision() >= 0x6000) || (theHeaderFormat->firmwareRevision() < 0x42D5)) {
        theFirmwareVersion = 2013;
        bool isGEM_fw = false;        /// Run3 ME11+GE11 OTMB fw with CCLUT and HMT
        bool isCCLUT_HMT_fw = false;  /// Run3 MEx1 OTMB fw with CCLUT and HMT
        bool isOTMB_Run2_fw = false;  /// Run2-compatible data format OTMB fw with Run3 revision code format
        bool isTMB_Run3_fw = false;
        bool isTMB_Run2_fw = false;
        bool isTMB_hybrid_fw = false;  /// Copper TMB hybrid fw Run2 CLCT + Run3 LCT/MPC + anode-only HMT (March 2022)
        bool isRun2_df = false;
        unsigned firmwareRevision = theHeaderFormat->firmwareRevision();
        /* There are OTMB2013 firmware versions exist, which reports firmwareRevision code = 0x0 */
        if ((firmwareRevision < 0x4000) && (firmwareRevision > 0x0)) { /* New (O)TMB firmware revision format */
          theFirmwareVersion = 2020;
          unsigned df_version = (firmwareRevision >> 9) & 0xF;  // 4-bits Data Format version
          unsigned major_ver = (firmwareRevision >> 5) & 0xF;   // 4-bits major version part
          // unsigned minor_ver = firmwareRevision & 0x1F;         // 5-bits minor version part
          switch (df_version) {
            case 0x4:
              isTMB_hybrid_fw = true;
              break;
            case 0x3:
              isGEM_fw = true;
              break;
            case 0x2:
              isCCLUT_HMT_fw = true;
              break;
            case 0x1:
              isOTMB_Run2_fw = true;
              break;
            case 0x0:
              if (major_ver == 1)
                isTMB_Run2_fw = true;
              else
                isTMB_Run3_fw = true;
              break;
            default:
              isGEM_fw = true;
          }
          if (major_ver == 1) {
            isRun2_df = true;
          }
        }
        if (theFirmwareVersion == 2020) {
          if (isGEM_fw) {
            if (isRun2_df) {
              theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_Run2(buf));
            } else {
              theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_GEM(buf));
            }
          } else if (isCCLUT_HMT_fw) {
            if (isRun2_df) {
              theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_Run2(buf));
            } else {
              theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_CCLUT(buf));
            }
          } else if (isOTMB_Run2_fw || isTMB_Run2_fw || isRun2_df) {
            theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_Run2(buf));
          } else if (isTMB_Run3_fw) {
            theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_CCLUT(buf));
          } else if (isTMB_hybrid_fw) {
            theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2020_TMB(buf));
          }

        } else {
          theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2013(buf));
        }

      } else {
        theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2007_rev0x50c3(buf));
      }
    }

  } else if (buf[0] == 0x6B0C) {
    theFirmwareVersion = 2006;
    theHeaderFormat = std::shared_ptr<CSCVTMBHeaderFormat>(new CSCTMBHeader2006(buf));
  } else {
    edm::LogError("CSCTMBHeader|CSCRawToDigi") << "failed to determine TMB firmware version!!";
  }
}

/*
void CSCTMBHeader::swapCLCTs(CSCCLCTDigi& digi1, CSCCLCTDigi& digi2)
{
  bool me11 = (theChamberId.station() == 1 &&
	       (theChamberId.ring() == 1 || theChamberId.ring() == 4));
  if (!me11) return;

  int cfeb1 = digi1.getCFEB();
  int cfeb2 = digi2.getCFEB();
  if (cfeb1 != cfeb2) return;

  bool me1a = (cfeb1 == 4);
  bool me1b = (cfeb1 != 4);
  bool zplus = (theChamberId.endcap() == 1);

  if ( (me1a && zplus) || (me1b && !zplus)) {
    // Swap CLCTs if they have the same quality and pattern # (priority
    // has to be given to the lower key).
    if (digi1.getQuality() == digi2.getQuality() &&
	digi1.getPattern() == digi2.getPattern()) {
      CSCCLCTDigi temp = digi1;
      digi1 = digi2;
      digi2 = temp;

      // Also re-number them.
      digi1.setTrknmb(1);
      digi2.setTrknmb(2);
    }
  }
}
*/

//FIXME Pick which LCT goes first
void CSCTMBHeader::add(const std::vector<CSCCLCTDigi> &digis) {
  // sort???
  if (!digis.empty()) {
    addCLCT0(digis[0]);
  }
  if (digis.size() > 1)
    addCLCT1(digis[1]);
}

void CSCTMBHeader::add(const std::vector<CSCCorrelatedLCTDigi> &digis) {
  // sort???
  if (!digis.empty())
    addCorrelatedLCT0(digis[0]);
  if (digis.size() > 1)
    addCorrelatedLCT1(digis[1]);
}

void CSCTMBHeader::add(const std::vector<CSCShowerDigi> &digis) {
  if (!digis.empty())
    theHeaderFormat->addShower(digis[0]);
}

CSCTMBHeader2007 CSCTMBHeader::tmbHeader2007() const {
  CSCTMBHeader2007 *result = dynamic_cast<CSCTMBHeader2007 *>(theHeaderFormat.get());
  if (result == nullptr) {
    throw cms::Exception("Could not get 2007 TMB header format");
  }
  return *result;
}

CSCTMBHeader2007_rev0x50c3 CSCTMBHeader::tmbHeader2007_rev0x50c3() const {
  CSCTMBHeader2007_rev0x50c3 *result = dynamic_cast<CSCTMBHeader2007_rev0x50c3 *>(theHeaderFormat.get());
  if (result == nullptr) {
    throw cms::Exception("Could not get 2007 rev0x50c3 TMB header format");
  }
  return *result;
}

CSCTMBHeader2013 CSCTMBHeader::tmbHeader2013() const {
  CSCTMBHeader2013 *result = dynamic_cast<CSCTMBHeader2013 *>(theHeaderFormat.get());
  if (result == nullptr) {
    throw cms::Exception("Could not get 2013 TMB header format");
  }
  return *result;
}

CSCTMBHeader2020_TMB CSCTMBHeader::tmbHeader2020_TMB() const {
  CSCTMBHeader2020_TMB *result = dynamic_cast<CSCTMBHeader2020_TMB *>(theHeaderFormat.get());
  if (result == nullptr) {
    throw cms::Exception("Could not get 2020 TMB Run3 header format");
  }
  return *result;
}

CSCTMBHeader2020_CCLUT CSCTMBHeader::tmbHeader2020_CCLUT() const {
  CSCTMBHeader2020_CCLUT *result = dynamic_cast<CSCTMBHeader2020_CCLUT *>(theHeaderFormat.get());
  if (result == nullptr) {
    throw cms::Exception("Could not get 2020 (O)TMB CCLUT header format");
  }
  return *result;
}

CSCTMBHeader2020_GEM CSCTMBHeader::tmbHeader2020_GEM() const {
  CSCTMBHeader2020_GEM *result = dynamic_cast<CSCTMBHeader2020_GEM *>(theHeaderFormat.get());
  if (result == nullptr) {
    throw cms::Exception("Could not get 2020 (O)TMB GEM header format");
  }
  return *result;
}

CSCTMBHeader2020_Run2 CSCTMBHeader::tmbHeader2020_Run2() const {
  CSCTMBHeader2020_Run2 *result = dynamic_cast<CSCTMBHeader2020_Run2 *>(theHeaderFormat.get());
  if (result == nullptr) {
    throw cms::Exception("Could not get 2020 (O)TMB legacy Run2 header format");
  }
  return *result;
}

CSCTMBHeader2006 CSCTMBHeader::tmbHeader2006() const {
  CSCTMBHeader2006 *result = dynamic_cast<CSCTMBHeader2006 *>(theHeaderFormat.get());
  if (result == nullptr) {
    throw cms::Exception("Could not get 2006 TMB header format");
  }
  return *result;
}

void CSCTMBHeader::selfTest(int firmwareVersion, int firmwareRevision) {
  constexpr bool debug = false;

  // tests packing and unpacking
  for (int station = 1; station <= 4; ++station) {
    for (int iendcap = 1; iendcap <= 2; ++iendcap) {
      CSCDetId detId(iendcap, station, 1, 1, 0);

      // the next-to-last is the BX, which only gets
      // saved in two bits and must be the same for clct0 and clct1.
      //CSCCLCTDigi clct0(1, 1, 4, 0, 0, 30, 3, 0, 1); // valid for 2006
      // In 2007 firmware, there are no distrips, so the 4th argument (strip
      // type) should always be set to 1 (halfstrips).
      CSCCLCTDigi clct0(1, 1, 4, 1, 0, 30, 4, 2, 1);  // valid for 2007
      CSCCLCTDigi clct1(1, 1, 3, 1, 1, 31, 1, 2, 2);

      // BX of LCT (8th argument) is 1-bit word (the least-significant bit
      // of ALCT's bx).
      CSCCorrelatedLCTDigi lct0(1, 1, 2, 10, 98, 5, 0, 1, 0, 0, 0, 0);
      CSCCorrelatedLCTDigi lct1(2, 1, 2, 20, 15, 9, 1, 0, 0, 0, 0, 0);

      // Use Run3 format digis for TMB firmwareVersion 2020
      // and revision codes for MEx1 CCLUT, ME11 CCLUT/GEM
      if (firmwareVersion >= 2020) {
        bool isGEM_fw = false;
        bool isCCLUT_HMT_fw = false;
        bool isOTMB_Run2_fw = false;
        bool isTMB_Run3_fw = false;
        bool isTMB_Run2_fw = false;
        bool isTMB_hybrid_fw = false;
        bool isRun2_df = false;
        unsigned df_version = (firmwareRevision >> 9) & 0xF;  // 4-bits Data Format version
        unsigned major_ver = (firmwareRevision >> 5) & 0xF;   // 4-bits major version part
        // unsigned minor_ver = firmwareRevision & 0x1F;         // 5-bits minor version part
        switch (df_version) {
          case 0x4:
            isTMB_hybrid_fw = true;
            break;
          case 0x3:
            isGEM_fw = true;
            break;
          case 0x2:
            isCCLUT_HMT_fw = true;
            break;
          case 0x1:
            isOTMB_Run2_fw = true;
            break;
          case 0x0:
            if (major_ver == 1)
              isTMB_Run2_fw = true;
            else
              isTMB_Run3_fw = true;
            break;
          default:
            isGEM_fw = true;
        }
        if (major_ver == 1) {
          isRun2_df = true;
        }
        if ((isGEM_fw || isCCLUT_HMT_fw || isTMB_Run3_fw) && !isRun2_df && !isOTMB_Run2_fw && !isTMB_Run2_fw &&
            !isTMB_hybrid_fw) {
          clct0 = CSCCLCTDigi(
              1, 6, 6, 1, 0, (120 % 32), (120 / 32), 2, 1, 3, 0xebf, CSCCLCTDigi::Version::Run3, true, false, 2, 6);
          clct1 = CSCCLCTDigi(
              1, 6, 3, 1, 1, (132 % 32), (132 / 32), 2, 2, 3, 0xe54, CSCCLCTDigi::Version::Run3, false, true, 1, 15);
        }
        if ((isGEM_fw || isCCLUT_HMT_fw || isTMB_Run3_fw) && !isRun2_df && !isOTMB_Run2_fw && !isTMB_Run2_fw &&
            !isTMB_hybrid_fw) {
          lct0 = CSCCorrelatedLCTDigi(
              1, 1, 3, 85, 120, 6, 0, 0, 0, 0, 0, 0, CSCCorrelatedLCTDigi::Version::Run3, true, false, 2, 6);
          lct1 = CSCCorrelatedLCTDigi(
              2, 1, 2, 81, 132, 3, 1, 0, 0, 0, 0, 0, CSCCorrelatedLCTDigi::Version::Run3, false, true, 0, 15);
        }
        if (isTMB_hybrid_fw) {
          lct0 = CSCCorrelatedLCTDigi(
              1, 1, 3, 85, 120, 6, 0, 0, 0, 0, 0, 0, CSCCorrelatedLCTDigi::Version::Run3, false, false, 0, 0);
          lct1 = CSCCorrelatedLCTDigi(
              2, 1, 2, 81, 132, 3, 1, 0, 0, 0, 0, 0, CSCCorrelatedLCTDigi::Version::Run3, false, false, 0, 0);
        }
      }

      CSCTMBHeader tmbHeader(firmwareVersion, firmwareRevision);
      tmbHeader.addCLCT0(clct0);
      tmbHeader.addCLCT1(clct1);
      tmbHeader.addCorrelatedLCT0(lct0);
      tmbHeader.addCorrelatedLCT1(lct1);
      std::vector<CSCCLCTDigi> clcts = tmbHeader.CLCTDigis(detId.rawId());
      // guess they got reordered
      assert(cscPackerCompare(clcts[0], clct0));
      assert(cscPackerCompare(clcts[1], clct1));
      if (debug) {
        std::cout << "Match for: " << clct0 << "\n";
        std::cout << "           " << clct1 << "\n \n";
      }

      std::vector<CSCCorrelatedLCTDigi> lcts = tmbHeader.CorrelatedLCTDigis(detId.rawId());
      assert(cscPackerCompare(lcts[0], lct0));
      assert(cscPackerCompare(lcts[1], lct1));
      if (debug) {
        std::cout << "Match for: " << lct0 << "\n";
        std::cout << "           " << lct1 << "\n";
      }

      // try packing and re-packing, to make sure they're the same
      unsigned short int *data = tmbHeader.data();
      CSCTMBHeader newHeader(data);
      clcts = newHeader.CLCTDigis(detId.rawId());
      assert(cscPackerCompare(clcts[0], clct0));
      assert(cscPackerCompare(clcts[1], clct1));
      lcts = newHeader.CorrelatedLCTDigis(detId.rawId());
      assert(cscPackerCompare(lcts[0], lct0));
      assert(cscPackerCompare(lcts[1], lct1));
    }
  }
}

std::ostream &operator<<(std::ostream &os, const CSCTMBHeader &hdr) {
  hdr.theHeaderFormat->print(os);
  return os;
}
