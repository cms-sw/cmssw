// \file AlignmentRcdChecker.cpp
//
// \author    : Marco Musich
// Revision   : $Revision: 1.1 $
// last update: $Date: 2022/05/11 14:44:00 $
// by         : $Author: musich $

// system includes
#include <string>
#include <map>
#include <vector>

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

  bool verbose_;
  bool compareStrict_;
  std::string label_;

  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> tkAliTokenRef_;
  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> tkAliTokenNew_;
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
    double meanX = 0;
    double rmsX = 0;
    double meanY = 0;
    double rmsY = 0;
    double meanZ = 0;
    double rmsZ = 0;
    double meanR = 0;
    double rmsR = 0;
    double dPhi;
    double meanPhi = 0;
    double rmsPhi = 0;

    std::vector<AlignTransform>::const_iterator iref = refAlignments->m_align.begin();
    for (std::vector<AlignTransform>::const_iterator i = alignments->m_align.begin(); i != alignments->m_align.end();
         ++i, ++iref) {
      meanX += i->translation().x() - iref->translation().x();
      rmsX += pow(i->translation().x() - iref->translation().x(), 2);

      meanY += i->translation().y() - iref->translation().y();
      rmsY += pow(i->translation().y() - iref->translation().y(), 2);

      meanZ += i->translation().z() - iref->translation().z();
      rmsZ += pow(i->translation().z() - iref->translation().z(), 2);

      meanR += i->translation().perp() - iref->translation().perp();
      rmsR += pow(i->translation().perp() - iref->translation().perp(), 2);

      dPhi = i->translation().phi() - iref->translation().phi();
      if (dPhi > M_PI)
        dPhi -= 2.0 * M_PI;
      if (dPhi < -M_PI)
        dPhi += 2.0 * M_PI;

      meanPhi += dPhi;
      rmsPhi += dPhi * dPhi;
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
                                     << std::setprecision(3) << meanX << " (RMS = " << sqrt(rmsX) << ")";
      edm::LogPrint("inspectRecord") << "    mean Y shift:   " << std::setw(12) << std::scientific
                                     << std::setprecision(3) << meanY << " (RMS = " << sqrt(rmsY) << ")";
      edm::LogPrint("inspectRecord") << "    mean Z shift:   " << std::setw(12) << std::scientific
                                     << std::setprecision(3) << meanZ << " (RMS = " << sqrt(rmsZ) << ")";
      edm::LogPrint("inspectRecord") << "    mean R shift:   " << std::setw(12) << std::scientific
                                     << std::setprecision(3) << meanR << " (RMS = " << sqrt(rmsR) << ")";
      edm::LogPrint("inspectRecord") << "    mean Phi shift: " << std::setw(12) << std::scientific
                                     << std::setprecision(3) << meanPhi << " (RMS = " << sqrt(rmsPhi) << ")";
    }  // verbose

    if (compareStrict_) {
      // do not let any of the coordinates to fluctuate less then 1um
      assert(meanX < 1e-4);
      assert(meanY < 1e-4);
      assert(meanZ < 1e-4);
      assert(meanR < 1e-4);
      assert(meanPhi < 1e-4);
    }

  } else {
    throw cms::Exception("Missing Input Data") << "Could not retrieve the input alignments to compare \n";
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlignmentRcdChecker);
