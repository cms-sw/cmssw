// \file AlignmentRcdScan.cpp
//
// \author    : Andreas Mussgiller
// Revision   : $Revision: 1.1 $
// last update: $Date: 2010/06/01 07:45:46 $
// by         : $Author: mussgill $

#include <string>
#include <map>
#include <vector>

#include <TMath.h>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"

#include "FWCore/Framework/interface/ESWatcher.h"

class AlignmentRcdScan : public edm::one::EDAnalyzer<> {
public:
  enum Mode { Unknown = 0, Tk = 1, DT = 2, CSC = 3 };

  explicit AlignmentRcdScan(const edm::ParameterSet& iConfig);
  ~AlignmentRcdScan();

  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);

private:
  void inspectRecord(const std::string& rcdname, const edm::Event& evt, const Alignments* alignments);

  int mode_;
  bool verbose_;

  edm::ESWatcher<TrackerAlignmentRcd> watchTk_;
  edm::ESWatcher<DTAlignmentRcd> watchDT_;
  edm::ESWatcher<CSCAlignmentRcd> watchCSC_;

  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> tkAliToken_;
  const edm::ESGetToken<Alignments, DTAlignmentRcd> dtAliToken_;
  const edm::ESGetToken<Alignments, CSCAlignmentRcd> cscAliToken_;

  Alignments* refAlignments_;
};

AlignmentRcdScan::AlignmentRcdScan(const edm::ParameterSet& iConfig)
    : verbose_(iConfig.getUntrackedParameter<bool>("verbose")),
      tkAliToken_(esConsumes()),
      dtAliToken_(esConsumes()),
      cscAliToken_(esConsumes()),
      refAlignments_(0) {
  std::string modestring = iConfig.getUntrackedParameter<std::string>("mode");
  if (modestring == "Tk") {
    mode_ = Tk;
  } else if (modestring == "DT") {
    mode_ = DT;
  } else if (modestring == "CSC") {
    mode_ = CSC;
  } else {
    mode_ = Unknown;
  }

  if (mode_ == Unknown) {
    throw cms::Exception("BadConfig") << "Mode " << modestring << " not known";
  }
}

AlignmentRcdScan::~AlignmentRcdScan() {
  if (refAlignments_)
    delete refAlignments_;
}

void AlignmentRcdScan::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  if (mode_ == Tk && watchTk_.check(evtSetup)) {
    const Alignments* alignments = &evtSetup.getData(tkAliToken_);
    inspectRecord("TrackerAlignmentRcd", evt, alignments);
  }
  if (mode_ == DT && watchDT_.check(evtSetup)) {
    const Alignments* alignments = &evtSetup.getData(dtAliToken_);
    inspectRecord("DTAlignmentRcd", evt, alignments);
  }
  if (mode_ == CSC && watchCSC_.check(evtSetup)) {
    const Alignments* alignments = &evtSetup.getData(cscAliToken_);
    inspectRecord("CSCAlignmentRcd", evt, alignments);
  }
}

void AlignmentRcdScan::inspectRecord(const std::string& rcdname, const edm::Event& evt, const Alignments* alignments) {
  edm::LogPrint("inspectRecord") << rcdname << " content starting from run " << evt.run();

  if (verbose_ == false) {
    edm::LogPrint("inspectRecord") << std::endl;
    return;
  }

  edm::LogPrint("inspectRecord") << " with " << alignments->m_align.size() << " entries" << std::endl;

  if (refAlignments_) {
    edm::LogPrint("inspectRecord") << "  Compared to previous record:" << std::endl;

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

    std::vector<AlignTransform>::const_iterator iref = refAlignments_->m_align.begin();
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

    edm::LogPrint("inspectRecord") << "    mean X shift:   " << std::setw(12) << std::scientific << std::setprecision(3)
                                   << meanX << " (RMS = " << sqrt(rmsX) << ")" << std::endl;
    edm::LogPrint("inspectRecord") << "    mean Y shift:   " << std::setw(12) << std::scientific << std::setprecision(3)
                                   << meanY << " (RMS = " << sqrt(rmsY) << ")" << std::endl;
    edm::LogPrint("inspectRecord") << "    mean Z shift:   " << std::setw(12) << std::scientific << std::setprecision(3)
                                   << meanZ << " (RMS = " << sqrt(rmsZ) << ")" << std::endl;
    edm::LogPrint("inspectRecord") << "    mean R shift:   " << std::setw(12) << std::scientific << std::setprecision(3)
                                   << meanR << " (RMS = " << sqrt(rmsR) << ")" << std::endl;
    edm::LogPrint("inspectRecord") << "    mean Phi shift: " << std::setw(12) << std::scientific << std::setprecision(3)
                                   << meanPhi << " (RMS = " << sqrt(rmsPhi) << ")" << std::endl;

    delete refAlignments_;
  }

  refAlignments_ = new Alignments(*alignments);

  edm::LogPrint("inspectRecord") << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlignmentRcdScan);
