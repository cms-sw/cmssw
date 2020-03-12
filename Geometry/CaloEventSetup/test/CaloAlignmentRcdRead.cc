#include <string>
#include <vector>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/General/interface/ClassName.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"

class CaloAlignmentRcdRead : public edm::one::EDAnalyzer<> {
public:
  explicit CaloAlignmentRcdRead(const edm::ParameterSet& /*iConfig*/)
      : ebToken_{esConsumes<Alignments, EBAlignmentRcd>(edm::ESInputTag{})},
        eeToken_{esConsumes<Alignments, EEAlignmentRcd>(edm::ESInputTag{})},
        esToken_{esConsumes<Alignments, ESAlignmentRcd>(edm::ESInputTag{})},
        nEventCalls_(0) {}
  ~CaloAlignmentRcdRead() override {}

  template <typename T>
  void dumpAlignments(const edm::EventSetup& evtSetup, edm::ESGetToken<Alignments, T>& token);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ESGetToken<Alignments, EBAlignmentRcd> ebToken_;
  edm::ESGetToken<Alignments, EEAlignmentRcd> eeToken_;
  edm::ESGetToken<Alignments, ESAlignmentRcd> esToken_;

  unsigned int nEventCalls_;
};

template <typename T>
void CaloAlignmentRcdRead::dumpAlignments(const edm::EventSetup& evtSetup, edm::ESGetToken<Alignments, T>& token) {
  const auto& alignments = evtSetup.getData(token);

  std::string recordName = Demangle(typeid(T).name())();

  LogDebug("CaloAlignmentRcdRead") << "Dumping alignments: " << recordName;

  for (const auto& i : alignments.m_align) {
    LogDebug("CaloAlignmentRcdRead") << "entry " << i.rawId() << " translation " << i.translation() << " angles "
                                     << i.rotation().eulerAngles();
  }
}

void CaloAlignmentRcdRead::analyze(const edm::Event& /*evt*/, const edm::EventSetup& evtSetup) {
  if (nEventCalls_ > 0) {
    edm::LogWarning("CaloAlignmentRcdRead") << "Reading from DB to be done only once, "
                                            << "set 'untracked PSet maxEvents = {untracked int32 input = 1}'.";

    return;
  }

  LogDebug("CaloAlignmentRcdRead") << "Reading from database in CaloAlignmentRcdRead::analyze...";

  dumpAlignments(evtSetup, ebToken_);
  dumpAlignments(evtSetup, eeToken_);
  dumpAlignments(evtSetup, esToken_);

  LogDebug("CaloAlignmentRcdRead") << "done!";

  nEventCalls_++;
}

DEFINE_FWK_MODULE(CaloAlignmentRcdRead);
