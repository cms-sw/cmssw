#ifndef Alignment_OfflineValidation_interface_DMRHelper_h
#define Alignment_OfflineValidation_interface_DMRHelper_h

// -*- C++ -*-
//
// Package:    Alignment/OfflineValidation
// Header:     DMRHelper
//
/**\class DMRHelper DMRHelper.h Alignment/OfflineValidation/interface/DMRHelper.h

 Description: building blocks shared between DMRChecker and FastDMRChecker
              (running mean/variance estimator, split-DMR booking, and
              by-layer/by-disk residual-and-pull booking).

 Implementation:
     Header-only on purpose: both consumers are EDAnalyzer plugins in the
     same package, so a template/inline-only library avoids adding a
     separate compiled library + BuildFile.xml <export> just for this.

     The two checkers run the exact same algorithms but want different
     histogram binning/ranges (FastDMRChecker: narrower ranges, tuned for
     quick turnaround; DMRChecker: wider ranges, tuned for full offline
     validation). Rather than duplicate each function once per binning
     choice, the binning is expressed as a `Traits` type and passed as a
     template parameter - a compile-time policy rather than a runtime
     branch or a pile of extra function arguments.
*/
//
// Original Author:  Marco Musich
//

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

#include <fmt/printf.h>

#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TH1D.h"
#include "TH1F.h"

namespace DMRHelper {

  // conversion factor used throughout for residuals: cm -> um
  constexpr float cmToUm = 10000.;

  //-----------------------------------------------------------------
  // Online (Welford) mean/variance accumulator, keyed by raw DetId.
  // Identical in DMRChecker and FastDMRChecker modulo field types
  // (int vs float for the direction flags) - float is used here since
  // it is a strict superset (no precision loss for the +/-1 values
  // that are ever stored in them).
  //-----------------------------------------------------------------
  struct Estimators {
    float rDirection = 0.f;
    float zDirection = 0.f;
    float rOrZDirection = 0.f;
    int hitCount = 0;
    float runningMeanOfRes_ = 0.f;
    float runningVarOfRes_ = 0.f;
    float runningNormMeanOfRes_ = 0.f;
    float runningNormVarOfRes_ = 0.f;
  };

  using EstimatorMap = std::map<uint32_t, Estimators>;

  //*************************************************************
  // Implementation of the online variance algorithm, as in
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
  //*************************************************************
  inline void updateOnlineMomenta(EstimatorMap& myDetails, uint32_t theID, float the_data, float the_pull) {
    auto& est = myDetails[theID];
    est.hitCount += 1;

    float delta = 0.f;
    float n_delta = 0.f;

    if (est.hitCount != 1) {
      delta = the_data - est.runningMeanOfRes_;
      n_delta = the_pull - est.runningNormMeanOfRes_;
      est.runningMeanOfRes_ += (delta / est.hitCount);
      est.runningNormMeanOfRes_ += (n_delta / est.hitCount);
    } else {
      est.runningMeanOfRes_ = the_data;
      est.runningNormMeanOfRes_ = the_pull;
    }

    float delta2 = the_data - est.runningMeanOfRes_;
    float n_delta2 = the_pull - est.runningNormMeanOfRes_;

    est.runningVarOfRes_ += delta * delta2;
    est.runningNormVarOfRes_ += n_delta * n_delta2;
  }

  //*************************************************************
  // Identify (subdet name, signed layer/disk number) for a module.
  // Used to key the by-layer / by-disk residual-and-pull histograms.
  //*************************************************************
  inline std::pair<std::string, int32_t> findSubdetAndLayer(uint32_t moduleID, const TrackerTopology* tTopo) {
    std::string subdet;
    int32_t layer = 0;
    const auto id = DetId(moduleID);
    switch (id.subdetId()) {
      // Pixel Barrel, Endcap
      case PixelSubdetector::PixelBarrel:
        subdet = "BPIX";
        layer = tTopo->pxbLayer(id);
        break;
      case PixelSubdetector::PixelEndcap:
        subdet = "FPIX";
        layer = tTopo->pxfDisk(id) * (tTopo->pxfSide(moduleID) == 1 ? -1 : +1);
        break;
      // Strip TIB, TID, TOB, TEC
      case StripSubdetector::TIB:
        subdet = "TIB";
        layer = tTopo->tibLayer(id);
        break;
      case StripSubdetector::TID:
        subdet = "TID";
        layer = tTopo->tidWheel(id) * (tTopo->tidSide(moduleID) == 1 ? -1 : +1);
        break;
      case StripSubdetector::TOB:
        subdet = "TOB";
        layer = tTopo->tobLayer(id);
        break;
      case StripSubdetector::TEC:
        subdet = "TEC";
        layer = tTopo->tecWheel(id) * (tTopo->tecSide(moduleID) == 1 ? -1 : +1);
        break;
      default:
        throw cms::Exception("Inconsistent Data") << "Unknown Tracker subdetector: " << id.subdetId();
    }
    return std::make_pair(subdet, layer);
  }

  //-----------------------------------------------------------------
  // Binning traits: the numbers that are allowed to differ between
  // consumers. FastDMRChecker uses DefaultDMRBinning as-is; DMRChecker
  // (wider DMRs, and pulls binned +/-3 instead of +/-5) specializes
  // with FullDMRBinning below. Add further specializations here if a
  // third consumer needs yet another set of ranges - the booking code
  // itself never has to change.
  //-----------------------------------------------------------------
  struct DefaultDMRBinning {
    static constexpr int splitDMRBins = 101;
    static constexpr double splitDMRMin = -50.5;
    static constexpr double splitDMRMax = 50.5;

    static constexpr int byLayerResBins = 100;
    static constexpr double byLayerResMin = -1000.;
    static constexpr double byLayerResMax = 1000.;

    static constexpr int byLayerPullBins = 100;
    static constexpr double byLayerPullMin = -5.;
    static constexpr double byLayerPullMax = 5.;
  };

  struct FullDMRBinning {
    static constexpr int splitDMRBins = 100;
    static constexpr double splitDMRMin = -200.;
    static constexpr double splitDMRMax = 200.;

    static constexpr int byLayerResBins = 100;
    static constexpr double byLayerResMin = -1000.;
    static constexpr double byLayerResMax = 1000.;

    static constexpr int byLayerPullBins = 100;
    static constexpr double byLayerPullMin = -3.;
    static constexpr double byLayerPullMax = 3.;
  };

  //*************************************************************
  // Generic booker of split DMRs (split by the sign of r- or
  // z-direction, depending on whether the module lives in a
  // barrel-like or endcap-like subdetector).
  //*************************************************************
  template <typename Traits = DefaultDMRBinning>
  std::array<TH1D*, 2> bookSplitDMRHistograms(const TFileDirectory& dir,
                                              const std::string& subdet,
                                              const std::string& vartype,
                                              bool isBarrel) {
    std::array<TH1D*, 2> out;
    const std::array<std::string, 2> sign_name = {{"plus", "minus"}};
    const std::array<std::string, 2> sign = {{">0", "<0"}};
    const std::string dirTag = isBarrel ? "rDir" : "zDir";

    for (unsigned int i = 0; i < 2; i++) {
      const std::string name_ = fmt::sprintf("DMR%s_%s_%s%s", subdet, vartype, dirTag, sign_name[i]);
      const std::string title_ = fmt::sprintf("Split DMR of %s-%s (%s%s)", subdet, vartype, dirTag, sign[i]);
      const std::string axisTitle_ = fmt::sprintf("mean of %s-residuals (%s%s);modules", vartype, dirTag, sign[i]);

      out[i] = dir.make<TH1D>(name_.c_str(),
                              fmt::sprintf("%s;%s", title_, axisTitle_).c_str(),
                              Traits::splitDMRBins,
                              Traits::splitDMRMin,
                              Traits::splitDMRMax);
    }
    return out;
  }

  //-----------------------------------------------------------------
  // Per-(subdet,layer) residual + pull histogram pair, booked lazily
  // on first use rather than pre-booked for a fixed number of
  // layers/disks - this sidesteps needing to know per-subdetector
  // layer counts up front (as DMRChecker's bookResidualsHistogram
  // does) at the cost of the booking happening mid-event the first
  // time a given key is seen.
  //-----------------------------------------------------------------
  struct HistoPair {
    TH1F* base = nullptr;
    TH1F* normed = nullptr;
  };
  struct HistoXY {
    HistoPair x;
    HistoPair y;
  };
  using HistoSet = std::map<std::pair<std::string, int32_t>, HistoXY>;

  //*************************************************************
  // Book (on first use) and fill the by-layer / by-disk residual
  // and pull histogram for a given (subdet, layer) key.
  //*************************************************************
  template <typename Traits = DefaultDMRBinning>
  void fillByLayer(HistoSet& set,
                   TFileDirectory& dir,
                   const std::pair<std::string, int32_t>& key,
                   bool isX,
                   float residual,
                   float pull) {
    HistoPair& hp = isX ? set[key].x : set[key].y;
    if (!hp.base) {
      const std::string coord = isX ? "X" : "Y";
      const std::string baseName = fmt::sprintf("h_%s_layer%d_Res%s", key.first, key.second, coord);
      const std::string baseTitle = fmt::sprintf(
          "%s (layer/disk %d) track %s-residuals;res_{%s'} [#mum];hits", key.first, key.second, coord, coord);
      hp.base = dir.make<TH1F>(
          baseName.c_str(), baseTitle.c_str(), Traits::byLayerResBins, Traits::byLayerResMin, Traits::byLayerResMax);

      const std::string normName = fmt::sprintf("h_%s_layer%d_Pull%s", key.first, key.second, coord);
      const std::string normTitle = fmt::sprintf("%s (layer/disk %d) track %s-pulls;res_{%s'}/#sigma_{res_{%s'}};hits",
                                                 key.first,
                                                 key.second,
                                                 coord,
                                                 coord,
                                                 coord);
      hp.normed = dir.make<TH1F>(
          normName.c_str(), normTitle.c_str(), Traits::byLayerPullBins, Traits::byLayerPullMin, Traits::byLayerPullMax);
    }

    hp.base->Fill(residual);
    hp.normed->Fill(pull);
  }

}  // namespace DMRHelper

#endif  // Alignment_OfflineValidation_interface_DMRHelper_h
