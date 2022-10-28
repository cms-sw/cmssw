
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>

#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondFormats/DataRecord/interface/PedestalsRcd.h"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TFile.h"

using namespace std;

namespace edmtest {
  class PedestalsAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
  public:
    explicit PedestalsAnalyzer(edm::ParameterSet const& p)
        : thePedestalToken_(esConsumes()), theBeginRunPedestalToken_(esConsumes<edm::Transition::BeginRun>()) {
      edm::LogPrint("PedestalsAnalyzer") << "PedestalsAnalyzer";
    }
    explicit PedestalsAnalyzer(int i) { edm::LogPrint("PedestalsAnalyzer") << "PedestalsAnalyzer " << i; }
    virtual ~PedestalsAnalyzer() { edm::LogPrint("PedestalsAnalyzer") << "~PedestalsAnalyzer "; }
    virtual void beginJob() override;
    virtual void beginRun(const edm::Run&, const edm::EventSetup& context) override;
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c) override;
    virtual void endRun(const edm::Run&, const edm::EventSetup&) override;

  private:
    const edm::ESGetToken<Pedestals, PedestalsRcd> thePedestalToken_, theBeginRunPedestalToken_;
  };
  void PedestalsAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& context) {
    edm::LogPrint("PedestalsAnalyzer") << "###PedestalsAnalyzer::beginRun";
    edm::LogPrint("PedestalsAnalyzer") << "got context";
    auto const& myBeginRunPed = &context.getData(theBeginRunPedestalToken_);
    edm::LogPrint("PedestalsAnalyzer") << "Pedestals* " << myBeginRunPed;
  }
  void PedestalsAnalyzer::beginJob() { edm::LogPrint("PedestalsAnalyzer") << "###PedestalsAnalyzer::beginJob"; }
  void PedestalsAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    edm::LogPrint("PedestalsAnalyzer") << " I AM IN RUN NUMBER " << e.id().run();
    edm::LogPrint("PedestalsAnalyzer") << " ---EVENT NUMBER " << e.id().event();
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("PedestalsRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogPrint("PedestalsAnalyzer") << "Record \"PedestalsRcd\" does not exist";
    }
    edm::LogPrint("PedestalsAnalyzer") << "got context";
    auto const& myped = &context.getData(thePedestalToken_);
    edm::LogPrint("PedestalsAnalyzer") << "Pedestals* " << myped;
    for (std::vector<Pedestals::Item>::const_iterator it = myped->m_pedestals.begin(); it != myped->m_pedestals.end();
         ++it)
      edm::LogPrint("PedestalsAnalyzer") << " mean: " << it->m_mean << " variance: " << it->m_variance;
    edm::LogPrint("PedestalsAnalyzer") << std::endl;

    TFile* f = TFile::Open("MyPedestal.xml", "recreate");
    f->WriteObjectAny(myped, "Pedestals", "Pedestals");
    f->Close();
  }
  void PedestalsAnalyzer::endRun(const edm::Run&, const edm::EventSetup&) {}
  DEFINE_FWK_MODULE(PedestalsAnalyzer);
}  // namespace edmtest
