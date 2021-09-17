#ifndef CONDCORE_RUNINFOPLUGINS_RUNINFOPAYLOADINSPECTORHELPER_H
#define CONDCORE_RUNINFOPLUGINS_RUNINFOPAYLOADINSPECTORHELPER_H

#include <vector>
#include <string>
#include <ctime>
#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// for the FED intervals
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#define MK_P(a, b) std::make_pair(a, b)

namespace RunInfoPI {

  enum state { k_fake = 0, k_valid = 1, k_invalid = 2 };

  // values are taken from https://github.com/cms-sw/cmssw/blob/master/MagneticField/GeomBuilder/plugins/VolumeBasedMagneticFieldESProducerFromDB.cc#L74-L75
  constexpr std::array<int, 7> nominalCurrents{{-1, 0, 9558, 14416, 16819, 18268, 19262}};
  constexpr std::array<float, 7> nominalFields{{3.8, 0., 2., 3., 3.5, 3.8, 4.}};

  // all parameter than can be displayed
  enum parameters {
    m_run,                   // int
    m_start_time_ll,         // long long;
    m_stop_time_ll,          // long long
    m_start_current,         // float
    m_stop_current,          // float
    m_avg_current,           // float
    m_max_current,           // float
    m_min_current,           // float
    m_run_interval_seconds,  // float
    m_fedIN,                 // unsigned int
    m_BField,                // float
    END_OF_TYPES
  };

  /************************************************/
  inline float theBField(const float current) {
    // logic is taken from https://github.com/cms-sw/cmssw/blob/master/MagneticField/GeomBuilder/plugins/VolumeBasedMagneticFieldESProducerFromDB.cc#L156

    int i = 0;
    for (; i < (int)nominalFields.size() - 1; i++) {
      if (2 * current < nominalCurrents[i] + nominalCurrents[i + 1]) {
        return nominalFields[i];
      }
    }
    return nominalFields[i];
  }

  /************************************************/
  inline float runDuration(const std::shared_ptr<RunInfo>& payload) {
    // calculation of the run duration in seconds
    time_t start_time = payload->m_start_time_ll;
    ctime(&start_time);
    time_t end_time = payload->m_stop_time_ll;
    ctime(&end_time);
    return difftime(end_time, start_time) / 1.0e+6;
  }

  /************************************************/
  inline std::string runStartTime(const std::shared_ptr<RunInfo>& payload) {
    const time_t start_time = payload->m_start_time_ll / 1.0e+6;
    return std::asctime(std::gmtime(&start_time));
  }

  /************************************************/
  inline std::string runEndTime(const std::shared_ptr<RunInfo>& payload) {
    const time_t end_time = payload->m_stop_time_ll / 1.0e+6;
    return std::asctime(std::gmtime(&end_time));
  }

  /************************************************/
  inline std::string getStringFromTypeEnum(const parameters& parameter) {
    switch (parameter) {
      case m_run:
        return "run number";
      case m_start_time_ll:
        return "start time";
      case m_stop_time_ll:
        return "stop time";
      case m_start_current:
        return "start current [A]";
      case m_stop_current:
        return "stop current [A]";
      case m_avg_current:
        return "average current [A]";
      case m_max_current:
        return "max current [A]";
      case m_min_current:
        return "min current [A]";
      case m_run_interval_seconds:
        return "run duration [s]";
      case m_fedIN:
        return "n. FEDs";
      case m_BField:
        return "B-field intensity [T]";
      default:
        return "should never be here";
    }
  }

  /************************************************/
  inline void reportSummaryMapPalette(TH2* obj) {
    static int pcol[20];

    float rgb[20][3];

    for (int i = 0; i < 20; i++) {
      if (i < 17) {
        rgb[i][0] = 0.80 + 0.01 * i;
        rgb[i][1] = 0.00 + 0.03 * i;
        rgb[i][2] = 0.00;
      } else if (i < 19) {
        rgb[i][0] = 0.80 + 0.01 * i;
        rgb[i][1] = 0.00 + 0.03 * i + 0.15 + 0.10 * (i - 17);
        rgb[i][2] = 0.00;
      } else if (i == 19) {
        rgb[i][0] = 0.00;
        rgb[i][1] = 0.80;
        rgb[i][2] = 0.00;
      }
      pcol[i] = TColor::GetColor(rgb[i][0], rgb[i][1], rgb[i][2]);
    }

    gStyle->SetPalette(20, pcol);

    if (obj) {
      obj->SetMinimum(-1.e-15);
      obj->SetMaximum(+1.0);
      obj->SetOption("colz");
    }
  }

  // FED bounds
  enum DET {
    SIPIXEL,
    SISTRIP,
    PRESHOWER,
    TOTEMRP_H,
    CTPPSDIAMOND,
    TOTEMRP_V,
    TOTEMRP_T,
    ECAL,
    CASTOR,
    HCAL,
    CSC,
    RPC,
    HCALPHASE1,
    SIPIXELPHASE1,
    GEM
  };

  using FEDMAP_T = std::map<RunInfoPI::DET, std::pair<int, int> >;

  inline FEDMAP_T buildFEDBounds() {
    static RunInfoPI::FEDMAP_T fb;
    fb.insert(MK_P(SIPIXEL, MK_P(FEDNumbering::MINSiPixelFEDID, FEDNumbering::MAXSiPixelFEDID)));
    fb.insert(MK_P(SISTRIP, MK_P(FEDNumbering::MINSiStripFEDID, FEDNumbering::MAXSiStripFEDID)));
    fb.insert(MK_P(PRESHOWER, MK_P(FEDNumbering::MINPreShowerFEDID, FEDNumbering::MAXPreShowerFEDID)));
    fb.insert(MK_P(TOTEMRP_H, MK_P(FEDNumbering::MINTotemRPHorizontalFEDID, FEDNumbering::MAXTotemRPHorizontalFEDID)));
    fb.insert(MK_P(CTPPSDIAMOND, MK_P(FEDNumbering::MINCTPPSDiamondFEDID, FEDNumbering::MAXCTPPSDiamondFEDID)));
    fb.insert(MK_P(TOTEMRP_V, MK_P(FEDNumbering::MINTotemRPVerticalFEDID, FEDNumbering::MAXTotemRPVerticalFEDID)));
    fb.insert(MK_P(TOTEMRP_T,
                   MK_P(FEDNumbering::MINTotemRPTimingVerticalFEDID, FEDNumbering::MAXTotemRPTimingVerticalFEDID)));
    fb.insert(MK_P(ECAL, MK_P(FEDNumbering::MINECALFEDID, FEDNumbering::MAXECALFEDID)));
    fb.insert(MK_P(CASTOR, MK_P(FEDNumbering::MINCASTORFEDID, FEDNumbering::MAXCASTORFEDID)));
    fb.insert(MK_P(HCAL, MK_P(FEDNumbering::MINHCALFEDID, FEDNumbering::MAXHCALFEDID)));
    fb.insert(MK_P(CSC, MK_P(FEDNumbering::MINCSCFEDID, FEDNumbering::MAXCSCFEDID)));
    fb.insert(MK_P(RPC, MK_P(FEDNumbering::MINRPCFEDID, FEDNumbering::MAXRPCFEDID)));
    fb.insert(MK_P(HCALPHASE1, MK_P(FEDNumbering::MINHCALuTCAFEDID, FEDNumbering::MAXHCALuTCAFEDID)));
    fb.insert(MK_P(SIPIXELPHASE1, MK_P(FEDNumbering::MINSiPixeluTCAFEDID, FEDNumbering::MAXSiPixeluTCAFEDID)));
    fb.insert(MK_P(GEM, MK_P(FEDNumbering::MINGEMFEDID, FEDNumbering::MAXGEMFEDID)));
    return fb;
  }
};  // namespace RunInfoPI
#endif
