#include <string>
#include <map>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "DataFormats/DetId/interface/DetId.h"

class GlobalPositionRcdRead : public edm::one::EDAnalyzer<> {
public:
  explicit GlobalPositionRcdRead(const edm::ParameterSet& iConfig) : GPRToken_(esConsumes()), nEventCalls_(0) {}
  ~GlobalPositionRcdRead() {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<Alignments, GlobalPositionRcd> GPRToken_;
  unsigned int nEventCalls_;
};

void GlobalPositionRcdRead::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (nEventCalls_ > 0) {
    edm::LogPrint("GlobalPositionRcdRead")
        << "Reading from DB to be done only once, "
        << "set 'untracked PSet maxEvents = {untracked int32 input = 1}'." << std::endl;

    return;
  }

  edm::LogPrint("GlobalPositionRcdRead") << "Reading from database in GlobalPositionRcdRead::analyze..." << std::endl;

  const Alignments* globalPositionRcd = &iSetup.getData(GPRToken_);

  edm::LogPrint("GlobalPositionRcdRead") << "Expecting entries in " << DetId(DetId::Tracker).rawId() << " "
                                         << DetId(DetId::Muon).rawId() << " " << DetId(DetId::Ecal).rawId() << " "
                                         << DetId(DetId::Hcal).rawId() << " " << DetId(DetId::Calo).rawId()
                                         << std::endl;
  for (std::vector<AlignTransform>::const_iterator i = globalPositionRcd->m_align.begin();
       i != globalPositionRcd->m_align.end();
       ++i) {
    edm::LogPrint("GlobalPositionRcdRead") << "entry " << i->rawId() << " translation " << i->translation()
                                           << " angles " << i->rotation().eulerAngles() << std::endl;
    edm::LogPrint("GlobalPositionRcdRead") << i->rotation() << std::endl;
  }

  edm::LogPrint("GlobalPositionRcdRead") << "done!" << std::endl;
  nEventCalls_++;
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalPositionRcdRead);
-- dummy change --
