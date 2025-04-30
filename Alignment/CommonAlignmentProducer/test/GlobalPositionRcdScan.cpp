#include <string>
#include <map>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/Framework/interface/ESWatcher.h"

#include "Alignment/CommonAlignment/interface/Utilities.h"

class GlobalPositionRcdScan : public edm::one::EDAnalyzer<> {
public:
  explicit GlobalPositionRcdScan(const edm::ParameterSet& iConfig);

  virtual ~GlobalPositionRcdScan() = default;
  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup);
  virtual void endJob();

private:
  edm::ESWatcher<GlobalPositionRcd> watcher_;
  const edm::ESGetToken<Alignments, GlobalPositionRcd> GPRToken_;

  bool eulerAngles_;
  bool alignAngles_;
  bool matrix_;
  edm::RunNumber_t firstRun_;
  edm::RunNumber_t lastRun_;
};

GlobalPositionRcdScan::GlobalPositionRcdScan(const edm::ParameterSet& iConfig)
    : GPRToken_(esConsumes()), eulerAngles_(false), alignAngles_(false), matrix_(false), firstRun_(0), lastRun_(0) {
  const std::string howRot(iConfig.getParameter<std::string>("rotation"));

  if (howRot == "euler" || howRot == "all")
    eulerAngles_ = true;
  if (howRot == "align" || howRot == "all")
    alignAngles_ = true;
  if (howRot == "matrix" || howRot == "all")
    matrix_ = true;

  if (!eulerAngles_ && !alignAngles_ && !matrix_) {
    edm::LogError("BadConfig") << "@SUB=GlobalPositionRcdScan"
                               << "Parameter 'rotation' should be 'euler', 'align',"
                               << "'matrix' or 'all', but is '" << howRot << "'. Treat as 'all'.";
    eulerAngles_ = alignAngles_ = matrix_ = true;
  }
}

void GlobalPositionRcdScan::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  lastRun_ = evt.run();
  if (0 == firstRun_)
    firstRun_ = lastRun_;

  if (watcher_.check(evtSetup)) {
    const Alignments* globalPositionRcd = &evtSetup.getData(GPRToken_);

    edm::LogPrint("GlobalPositionRcdScan")
        << "=====================================================\n"
        << "GlobalPositionRcd content starting from run " << evt.run() << ":" << std::endl;

    for (std::vector<AlignTransform>::const_iterator i = globalPositionRcd->m_align.begin();
         i != globalPositionRcd->m_align.end();
         ++i) {
      edm::LogPrint("GlobalPositionRcdScan") << "  Component ";
      if (i->rawId() == DetId(DetId::Tracker).rawId()) {
        edm::LogPrint("GlobalPositionRcdScan") << "Tracker";
      } else if (i->rawId() == DetId(DetId::Muon).rawId()) {
        edm::LogPrint("GlobalPositionRcdScan") << "Muon   ";
      } else if (i->rawId() == DetId(DetId::Ecal).rawId()) {
        edm::LogPrint("GlobalPositionRcdScan") << "Ecal   ";
      } else if (i->rawId() == DetId(DetId::Hcal).rawId()) {
        edm::LogPrint("GlobalPositionRcdScan") << "Hcal   ";
      } else if (i->rawId() == DetId(DetId::Calo).rawId()) {
        edm::LogPrint("GlobalPositionRcdScan") << "Calo   ";
      } else {
        edm::LogPrint("GlobalPositionRcdScan") << "Unknown";
      }
      edm::LogPrint("GlobalPositionRcdScan")
          << " entry " << i->rawId() << "\n     translation " << i->translation() << "\n";
      const AlignTransform::Rotation hepRot(i->rotation());
      if (eulerAngles_) {
        edm::LogPrint("GlobalPositionRcdScan") << "     euler angles " << hepRot.eulerAngles() << std::endl;
      }
      if (alignAngles_) {
        const align::RotationType matrix(hepRot.xx(),
                                         hepRot.xy(),
                                         hepRot.xz(),
                                         hepRot.yx(),
                                         hepRot.yy(),
                                         hepRot.yz(),
                                         hepRot.zx(),
                                         hepRot.zy(),
                                         hepRot.zz());
        const AlgebraicVector angles(align::toAngles(matrix));
        edm::LogPrint("GlobalPositionRcdScan")
            << "     alpha, beta, gamma (" << angles[0] << ", " << angles[1] << ", " << angles[2] << ')' << std::endl;
      }
      if (matrix_) {
        edm::LogPrint("GlobalPositionRcdScan") << "     rotation matrix " << hepRot << std::endl;
      }
    }
  }
}

void GlobalPositionRcdScan::endJob() {
  edm::LogPrint("GlobalPositionRcdScan") << "\n=====================================================\n"
                                         << "=====================================================\n"
                                         << "Checked run range " << firstRun_ << " to " << lastRun_ << "." << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalPositionRcdScan);
-- dummy change --
