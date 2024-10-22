#ifndef SiStripHitEfficiencyHelpers_H
#define SiStripHitEfficiencyHelpers_H

// A bunch of helper functions to deal with menial tasks in the
// hit efficiency computation for the PCL workflow

// system includes
#include <fmt/printf.h>
#include <string>

// user includes
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

// ROOT includes
#include "TEfficiency.h"
#include "TProfile.h"
#include "TString.h"

namespace {

  enum bounds {
    k_LayersStart = 0,
    k_LayersAtTIBEnd = 4,
    k_LayersAtTOBEnd = 10,
    k_LayersAtTIDEnd = 13,
    k_LayersAtTECEnd = 22,
    k_END_OF_LAYERS = 23,
    k_END_OF_LAYS_AND_RINGS = 35
  };

  /*
   * for the trend plots of efficiency vs some variable
   */
  enum projections { k_vs_LUMI = 0, k_vs_PU = 1, k_vs_BX = 2, k_SIZE = 3 };

  const std::array<std::string, projections::k_SIZE> projFolder = {{"VsLumi", "VsPu", "VsBx"}};
  const std::array<std::string, projections::k_SIZE> projFoundHisto = {
      {"layerfound_vsLumi_layer_", "layerfound_vsPU_layer_", "foundVsBx_layer"}};
  const std::array<std::string, projections::k_SIZE> projTotalHisto = {
      {"layertotal_vsLumi_layer_", "layertotal_vsPU_layer_", "totalVsBx_layer"}};
  const std::array<std::string, projections::k_SIZE> projTitle = {{"Inst Lumi", "Pile-Up", "Bunch Crossing"}};
  const std::array<std::string, projections::k_SIZE> projXtitle = {
      {"instantaneous luminosity [Hz/cm^{2}]", "Pile-Up events", "Bunch Crossing number"}};

  inline void replaceInString(std::string& str, const std::string& from, const std::string& to) {
    if (from.empty())
      return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
      str.replace(start_pos, from.length(), to);
      start_pos += to.length();  // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
  }

  inline unsigned int checkLayer(unsigned int iidd, const TrackerTopology* tTopo) {
    switch (DetId(iidd).subdetId()) {
      case SiStripSubdetector::TIB:
        return tTopo->tibLayer(iidd);
      case SiStripSubdetector::TOB:
        return tTopo->tobLayer(iidd) + bounds::k_LayersAtTIBEnd;
      case SiStripSubdetector::TID:
        return tTopo->tidWheel(iidd) + bounds::k_LayersAtTOBEnd;
      case SiStripSubdetector::TEC:
        return tTopo->tecWheel(iidd) + bounds::k_LayersAtTIDEnd;
      default:
        return bounds::k_LayersStart;
    }
  }

  inline std::string layerName(unsigned int k, const bool showRings, const unsigned int nTEClayers) {
    const std::string ringlabel{showRings ? "R" : "D"};
    if (k > bounds::k_LayersStart && k <= bounds::k_LayersAtTIBEnd) {
      return fmt::format("TIB L{:d}", k);
    } else if (k > bounds::k_LayersAtTIBEnd && k <= bounds::k_LayersAtTOBEnd) {
      return fmt::format("TOB L{:d}", k - bounds::k_LayersAtTIBEnd);
    } else if (k > bounds::k_LayersAtTOBEnd && k <= bounds::k_LayersAtTIDEnd) {
      return fmt::format("TID {0}{1:d}", ringlabel, k - bounds::k_LayersAtTOBEnd);
    } else if (k > bounds::k_LayersAtTIDEnd && k <= bounds::k_LayersAtTIDEnd + nTEClayers) {
      return fmt::format("TEC {0}{1:d}", ringlabel, k - bounds::k_LayersAtTIDEnd);
    } else {
      return "should never be here!";
    }
  }

  inline std::string layerSideName(Long_t k, const bool showRings, const unsigned int nTEClayers) {
    const std::string ringlabel{showRings ? "R" : "D"};
    if (k > bounds::k_LayersStart && k <= bounds::k_LayersAtTIBEnd) {
      return fmt::format("TIB L{:d}", k);
    } else if (k > bounds::k_LayersAtTIBEnd && k <= bounds::k_LayersAtTOBEnd) {
      return fmt::format("TOB L{:d}", k - bounds::k_LayersAtTIBEnd);
    } else if (k > bounds::k_LayersAtTOBEnd && k < 14) {
      return fmt::format("TID- {0}{1:d}", ringlabel, k - bounds::k_LayersAtTOBEnd);
    } else if (k > 13 && k < 17) {
      return fmt::format("TID+ {0}{1:d}", ringlabel, k - 13);
    } else if (k > 16 && k < 17 + nTEClayers) {
      return fmt::format("TEC- {0}{1:d}", ringlabel, k - 16);
    } else if (k > 16 + nTEClayers) {
      return fmt::format("TEC+ {0}{1:d}", ringlabel, k - 16 - nTEClayers);
    } else {
      return "shoud never be here!";
    }
  }

  inline double checkConsistency(const StripClusterParameterEstimator::LocalValues& parameters,
                                 double xx,
                                 double xerr) {
    double error = sqrt(parameters.second.xx() + xerr * xerr);
    double separation = abs(parameters.first.x() - xx);
    double consistency = separation / error;
    return consistency;
  }

  inline bool isDoubleSided(unsigned int iidd, const TrackerTopology* tTopo) {
    unsigned int layer;
    switch (DetId(iidd).subdetId()) {
      case SiStripSubdetector::TIB:
        layer = tTopo->tibLayer(iidd);
        return (layer == 1 || layer == 2);
      case SiStripSubdetector::TOB:
        layer = tTopo->tobLayer(iidd) + 4;
        return (layer == 5 || layer == 6);
      case SiStripSubdetector::TID:
        layer = tTopo->tidRing(iidd) + 10;
        return (layer == 11 || layer == 12);
      case SiStripSubdetector::TEC:
        layer = tTopo->tecRing(iidd) + 13;
        return (layer == 14 || layer == 15 || layer == 18);
      default:
        return false;
    }
  }

  inline bool check2DPartner(unsigned int iidd, const std::vector<TrajectoryMeasurement>& traj) {
    unsigned int partner_iidd = 0;
    bool found2DPartner = false;
    // first get the id of the other detector
    if ((iidd & 0x3) == 1)
      partner_iidd = iidd + 1;
    if ((iidd & 0x3) == 2)
      partner_iidd = iidd - 1;
    // next look in the trajectory measurements for a measurement from that detector
    // loop through trajectory measurements to find the partner_iidd
    for (const auto& tm : traj) {
      if (tm.recHit()->geographicalId().rawId() == partner_iidd) {
        found2DPartner = true;
      }
    }
    return found2DPartner;
  }

  inline bool isInBondingExclusionZone(
      unsigned int iidd, unsigned int TKlayers, double yloc, double yErr, const TrackerTopology* tTopo) {
    constexpr float exclusionWidth = 0.4;
    constexpr float TOBexclusion = 0.0;
    constexpr float TECexRing5 = -0.89;
    constexpr float TECexRing6 = -0.56;
    constexpr float TECexRing7 = 0.60;

    //Added by Chris Edelmaier to do TEC bonding exclusion
    const int subdetector = ((iidd >> 25) & 0x7);
    const int ringnumber = ((iidd >> 5) & 0x7);

    bool inZone = false;
    //New TOB and TEC bonding region exclusion zone
    if ((TKlayers >= 5 && TKlayers < 11) || ((subdetector == 6) && ((ringnumber >= 5) && (ringnumber <= 7)))) {
      //There are only 2 cases that we need to exclude for
      float highzone = 0.0;
      float lowzone = 0.0;
      float higherr = yloc + 5.0 * yErr;
      float lowerr = yloc - 5.0 * yErr;
      if (TKlayers >= 5 && TKlayers < 11) {
        //TOB zone
        highzone = TOBexclusion + exclusionWidth;
        lowzone = TOBexclusion - exclusionWidth;
      } else if (ringnumber == 5) {
        //TEC ring 5
        highzone = TECexRing5 + exclusionWidth;
        lowzone = TECexRing5 - exclusionWidth;
      } else if (ringnumber == 6) {
        //TEC ring 6
        highzone = TECexRing6 + exclusionWidth;
        lowzone = TECexRing6 - exclusionWidth;
      } else if (ringnumber == 7) {
        //TEC ring 7
        highzone = TECexRing7 + exclusionWidth;
        lowzone = TECexRing7 - exclusionWidth;
      }
      //Now that we have our exclusion region, we just have to properly identify it
      if ((highzone <= higherr) && (highzone >= lowerr))
        inZone = true;
      if ((lowzone >= lowerr) && (lowzone <= higherr))
        inZone = true;
      if ((higherr <= highzone) && (higherr >= lowzone))
        inZone = true;
      if ((lowerr >= lowzone) && (lowerr <= highzone))
        inZone = true;
    }
    return inZone;
  }

  struct ClusterInfo {
    float xResidual;
    float xResidualPull;
    float xLocal;
    ClusterInfo(float xRes, float xResPull, float xLoc) : xResidual(xRes), xResidualPull(xResPull), xLocal(xLoc) {}
  };

  inline float calcPhi(float x, float y) {
    float phi = 0;
    if ((x >= 0) && (y >= 0))
      phi = std::atan(y / x);
    else if ((x >= 0) && (y <= 0))
      phi = std::atan(y / x) + 2 * M_PI;
    else if ((x <= 0) && (y >= 0))
      phi = std::atan(y / x) + M_PI;
    else
      phi = std::atan(y / x) + M_PI;
    phi = phi * 180.0 / M_PI;

    return phi;
  }

  inline TProfile* computeEff(const TH1F* num, const TH1F* denum, const std::string nameHist) {
    std::string name = "eff_" + nameHist;
    std::string title = "SiStrip Hit Efficiency" + std::string(num->GetTitle());
    TProfile* efficHist = new TProfile(name.c_str(),
                                       title.c_str(),
                                       denum->GetXaxis()->GetNbins(),
                                       denum->GetXaxis()->GetXmin(),
                                       denum->GetXaxis()->GetXmax());

    for (int i = 1; i <= denum->GetNbinsX(); i++) {
      double nNum = num->GetBinContent(i);
      double nDenum = denum->GetBinContent(i);
      if (nDenum == 0 || nNum == 0) {
        continue;
      }
      if (nNum > nDenum) {
        edm::LogWarning("SiStripHitEfficiencyHelpers")
            << "Alert! specific bin's num is bigger than denum " << i << " " << nNum << " " << nDenum;
        nNum = nDenum;  // set the efficiency to 1
      }
      const double effVal = nNum / nDenum;
      efficHist->SetBinContent(i, effVal);
      efficHist->SetBinEntries(i, 1);
      const double errLo = TEfficiency::ClopperPearson((int)nDenum, (int)nNum, 0.683, false);
      const double errUp = TEfficiency::ClopperPearson((int)nDenum, (int)nNum, 0.683, true);
      const double errVal = (effVal - errLo > errUp - effVal) ? effVal - errLo : errUp - effVal;
      efficHist->SetBinError(i, sqrt(effVal * effVal + errVal * errVal));

      LogDebug("SiStripHitEfficiencyHelpers") << __PRETTY_FUNCTION__ << " " << nameHist << " bin:" << i
                                              << " err:" << sqrt(effVal * effVal + errVal * errVal);
    }
    return efficHist;
  }

  inline Measurement1D computeCPEfficiency(const double num, const double den) {
    if (den > 0) {
      const double effVal = num / den;
      const double errLo = TEfficiency::ClopperPearson((int)den, (int)num, 0.683, false);
      const double errUp = TEfficiency::ClopperPearson((int)den, (int)num, 0.683, true);
      const double errVal = (effVal - errLo > errUp - effVal) ? effVal - errLo : errUp - effVal;
      return Measurement1D(effVal, errVal);
    } else {
      return Measurement1D(0., 0.);
    }
  }
}  // namespace
#endif
