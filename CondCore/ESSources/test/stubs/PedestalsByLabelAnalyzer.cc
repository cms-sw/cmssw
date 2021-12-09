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

using namespace std;

namespace edmtest {
  class PedestalsByLabelAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit PedestalsByLabelAnalyzer(edm::ParameterSet const& p)
        : thePedestalToken_(esConsumes(edm::ESInputTag("", "lab3d"))) {
      edm::LogPrint("PedestalsByLabelAnalyzer") << "PedestalsByLabelAnalyzer";
    }
    explicit PedestalsByLabelAnalyzer(int i) {
      edm::LogPrint("PedestalsByLabelAnalyzer") << "PedestalsByLabelAnalyzer " << i;
    }
    virtual ~PedestalsByLabelAnalyzer() { edm::LogPrint("PedestalsByLabelAnalyzer") << "~PedestalsByLabelAnalyzer "; }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    const edm::ESGetToken<Pedestals, PedestalsRcd> thePedestalToken_;
  };

  void PedestalsByLabelAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    edm::LogPrint("PedestalsByLabelAnalyzer") << " I AM IN RUN NUMBER " << e.id().run();
    edm::LogPrint("PedestalsByLabelAnalyzer") << " ---EVENT NUMBER " << e.id().event();
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("PedestalsRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogPrint("PedestalsByLabelAnalyzer") << "Record \"PedestalsRcd\" does not exist";
    }
    edm::LogPrint("PedestalsByLabelAnalyzer") << "got context";
    auto const& myped = &context.getData(thePedestalToken_);
    edm::LogPrint("PedestalsByLabelAnalyzer") << "Pedestals* " << myped;
    for (std::vector<Pedestals::Item>::const_iterator it = myped->m_pedestals.begin(); it != myped->m_pedestals.end();
         ++it)
      edm::LogPrint("PedestalsByLabelAnalyzer") << " mean: " << it->m_mean << " variance: " << it->m_variance;
    edm::LogPrint("PedestalsByLabelAnalyzer") << std::endl;
  }
  DEFINE_FWK_MODULE(PedestalsByLabelAnalyzer);
}  // namespace edmtest
