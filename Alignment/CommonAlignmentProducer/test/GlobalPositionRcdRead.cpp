#include <string>
#include <map>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "DataFormats/DetId/interface/DetId.h"

class GlobalPositionRcdRead : public edm::EDAnalyzer {
public:
  explicit GlobalPositionRcdRead(const edm::ParameterSet& iConfig) : nEventCalls_(0) {}
  ~GlobalPositionRcdRead() override {}
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;

private:
  unsigned int nEventCalls_;
};

void GlobalPositionRcdRead::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  if (nEventCalls_ > 0) {
    std::cout << "Reading from DB to be done only once, "
              << "set 'untracked PSet maxEvents = {untracked int32 input = 1}'." << std::endl;

    return;
  }

  std::cout << "Reading from database in GlobalPositionRcdRead::analyze..." << std::endl;

  edm::ESHandle<Alignments> globalPositionRcd;
  evtSetup.get<GlobalPositionRcd>().get(globalPositionRcd);

  std::cout << "Expecting entries in " << DetId(DetId::Tracker).rawId() << " " << DetId(DetId::Muon).rawId() << " "
            << DetId(DetId::Ecal).rawId() << " " << DetId(DetId::Hcal).rawId() << " " << DetId(DetId::Calo).rawId()
            << std::endl;
  for (std::vector<AlignTransform>::const_iterator i = globalPositionRcd->m_align.begin();
       i != globalPositionRcd->m_align.end();
       ++i) {
    std::cout << "entry " << i->rawId() << " translation " << i->translation() << " angles "
              << i->rotation().eulerAngles() << std::endl;
    std::cout << i->rotation() << std::endl;
  }

  std::cout << "done!" << std::endl;
  nEventCalls_++;
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalPositionRcdRead);
