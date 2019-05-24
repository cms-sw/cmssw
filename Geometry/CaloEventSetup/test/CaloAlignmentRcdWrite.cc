#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Utilities/General/interface/ClassName.h"

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"

#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"

class CaloAlignmentRcdWrite : public edm::one::EDAnalyzer<> {
public:
  explicit CaloAlignmentRcdWrite(const edm::ParameterSet& /*iConfig*/)
      : ebToken_{esConsumes<Alignments, EBAlignmentRcd>(edm::ESInputTag{})},
        eeToken_{esConsumes<Alignments, EEAlignmentRcd>(edm::ESInputTag{})},
        esToken_{esConsumes<Alignments, ESAlignmentRcd>(edm::ESInputTag{})},
        nEventCalls_(0) {}
  ~CaloAlignmentRcdWrite() override {}

  template <typename T>
  void writeAlignments(const edm::EventSetup& evtSetup, edm::ESGetToken<Alignments, T>& token);

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
void CaloAlignmentRcdWrite::writeAlignments(const edm::EventSetup& evtSetup, edm::ESGetToken<Alignments, T>& token) {
  const auto& alignmentsES = evtSetup.getData(token);

  std::string recordName = Demangle(typeid(T).name())();

  std::cout << "Uploading alignments to the database: " << recordName << std::endl;

  edm::Service<cond::service::PoolDBOutputService> poolDbService;

  if (!poolDbService.isAvailable())
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  Alignments* alignments = new Alignments(alignmentsES);

  poolDbService->writeOne<Alignments>(&(*alignments), poolDbService->currentTime(), recordName);
}

void CaloAlignmentRcdWrite::analyze(const edm::Event& /*evt*/, const edm::EventSetup& evtSetup) {
  if (nEventCalls_ > 0) {
    std::cout << "Writing to DB to be done only once, "
              << "set 'untracked PSet maxEvents = {untracked int32 input = 1}'."
              << "(Your writing should be fine.)" << std::endl;
    return;
  }

  writeAlignments<EBAlignmentRcd>(evtSetup, ebToken_);
  writeAlignments<EEAlignmentRcd>(evtSetup, eeToken_);
  writeAlignments<ESAlignmentRcd>(evtSetup, esToken_);

  std::cout << "done!" << std::endl;
  nEventCalls_++;
}

DEFINE_FWK_MODULE(CaloAlignmentRcdWrite);
