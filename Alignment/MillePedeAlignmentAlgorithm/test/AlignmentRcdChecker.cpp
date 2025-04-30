// \file AlignmentRcdChecker.cpp
//
// \author    : Marco Musich
// Revision   : $Revision: 1.1 $
// last update: $Date: 2022/05/11 14:44:00 $
// by         : $Author: musich $

// system includes
#include <string>
#include <map>
#include <ranges>
#include <vector>
#include <tuple>
#include <iterator>

// user includes
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

// ROOT includes
#include <TMath.h>

namespace checker {
  template <typename T1, typename T2>
  auto zip(const std::vector<T1>& v1, const std::vector<T2>& v2) {
    std::vector<std::tuple<T1, T2>> result;

    auto minSize = std::min(v1.size(), v2.size());
    for (size_t i = 0; i < minSize; ++i) {
      result.emplace_back(v1[i], v2[i]);
    }

    return result;
  }
}  // namespace checker

class AlignmentRcdChecker : public edm::one::EDAnalyzer<> {
public:
  explicit AlignmentRcdChecker(const edm::ParameterSet& iConfig);
  ~AlignmentRcdChecker() = default;

  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);

private:
  void inspectRecord(const std::string& rcdname,
                     const edm::Event& evt,
                     const Alignments* refAlignments,
                     const Alignments* alignments);

  const bool verbose_;
  const bool compareStrict_;
  const std::string label_;

  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> tkAliTokenRef_;
  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> tkAliTokenNew_;

  static constexpr double strictTolerance_ = 1e-4;  // if in cm (i.e. 1 micron)
};

AlignmentRcdChecker::AlignmentRcdChecker(const edm::ParameterSet& iConfig)
    : verbose_(iConfig.getParameter<bool>("verbose")),
      compareStrict_(iConfig.getParameter<bool>("compareStrict")),
      label_(iConfig.getParameter<std::string>("label")),
      tkAliTokenRef_(esConsumes()),
      tkAliTokenNew_(esConsumes(edm::ESInputTag{"", label_})) {}

void AlignmentRcdChecker::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  const Alignments* alignmentsRef = &evtSetup.getData(tkAliTokenRef_);
  const Alignments* alignmentsNew = &evtSetup.getData(tkAliTokenNew_);
  inspectRecord("TrackerAlignmentRcd", evt, alignmentsRef, alignmentsNew);
}

void AlignmentRcdChecker::inspectRecord(const std::string& rcdname,
                                        const edm::Event& evt,
                                        const Alignments* refAlignments,
                                        const Alignments* alignments) {
  edm::LogPrint("inspectRecord") << rcdname << " content starting from run " << evt.run();
  edm::LogPrint("inspectRecord") << " with " << alignments->m_align.size() << " entries";

  if (refAlignments && alignments) {
    double meanX = 0, rmsX = 0;
    double meanY = 0, rmsY = 0;
    double meanZ = 0, rmsZ = 0;
    double meanR = 0, rmsR = 0;
    double dPhi, meanPhi = 0, rmsPhi = 0;

    for (const auto& [alignment, refAlignment] : checker::zip(alignments->m_align, refAlignments->m_align)) {
      auto delta = alignment.translation() - refAlignment.translation();

      meanX += delta.x();
      rmsX += pow(delta.x(), 2);

      meanY += delta.y();
      rmsY += pow(delta.y(), 2);

      meanZ += delta.z();
      rmsZ += pow(delta.z(), 2);

      meanR += delta.perp();
      rmsR += pow(delta.perp(), 2);

      dPhi = alignment.translation().phi() - refAlignment.translation().phi();
      if (dPhi > M_PI)
        dPhi -= 2.0 * M_PI;
      if (dPhi < -M_PI)
        dPhi += 2.0 * M_PI;

      meanPhi += dPhi;
      rmsPhi += std::pow(dPhi, 2);
    }

    meanX /= alignments->m_align.size();
    rmsX /= alignments->m_align.size();
    meanY /= alignments->m_align.size();
    rmsY /= alignments->m_align.size();
    meanZ /= alignments->m_align.size();
    rmsZ /= alignments->m_align.size();
    meanR /= alignments->m_align.size();
    rmsR /= alignments->m_align.size();
    meanPhi /= alignments->m_align.size();
    rmsPhi /= alignments->m_align.size();

    if (verbose_) {
      edm::LogPrint("inspectRecord") << "  Compared to previous record:";
      edm::LogPrint("inspectRecord") << "    mean X shift:   " << std::setw(12) << std::scientific
                                     << std::setprecision(3) << meanX << "  [cm] (RMS = " << sqrt(rmsX) << " [cm])";
      edm::LogPrint("inspectRecord") << "    mean Y shift:   " << std::setw(12) << std::scientific
                                     << std::setprecision(3) << meanY << "  [cm] (RMS = " << sqrt(rmsY) << " [cm])";
      edm::LogPrint("inspectRecord") << "    mean Z shift:   " << std::setw(12) << std::scientific
                                     << std::setprecision(3) << meanZ << "  [cm] (RMS = " << sqrt(rmsZ) << " [cm])";
      edm::LogPrint("inspectRecord") << "    mean R shift:   " << std::setw(12) << std::scientific
                                     << std::setprecision(3) << meanR << "  [cm] (RMS = " << sqrt(rmsR) << " [cm])";
      edm::LogPrint("inspectRecord") << "    mean Phi shift: " << std::setw(12) << std::scientific
                                     << std::setprecision(3) << meanPhi << " [rad] (RMS = " << sqrt(rmsPhi)
                                     << " [rad])";
    }  // verbose

    if (compareStrict_) {
      // do not let any of the coordinates to fluctuate less then 1um
      assert(meanX < strictTolerance_);    // 1 micron
      assert(meanY < strictTolerance_);    // 1 micron
      assert(meanZ < strictTolerance_);    // 1 micron
      assert(meanR < strictTolerance_);    // 1 micron
      assert(meanPhi < strictTolerance_);  // 10 micro-rad
    }

  } else {
    throw cms::Exception("Missing Input Data") << "Could not retrieve the input alignments to compare \n";
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlignmentRcdChecker);
-- dummy change --
