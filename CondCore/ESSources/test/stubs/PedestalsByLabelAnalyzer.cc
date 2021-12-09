/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>

#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondFormats/DataRecord/interface/PedestalsRcd.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;

namespace edmtest {
  class PedestalsByLabelAnalyzer : public edm::EDAnalyzer {
  public:
    explicit PedestalsByLabelAnalyzer(edm::ParameterSet const& p)
        : thePedestalToken_(esConsumes(edm::ESInputTag("", "lab3d"))) {
      std::cout << "PedestalsByLabelAnalyzer" << std::endl;
    }
    explicit PedestalsByLabelAnalyzer(int i) { std::cout << "PedestalsByLabelAnalyzer " << i << std::endl; }
    virtual ~PedestalsByLabelAnalyzer() { std::cout << "~PedestalsByLabelAnalyzer " << std::endl; }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    const edm::ESGetToken<Pedestals, PedestalsRcd> thePedestalToken_;
  };

  void PedestalsByLabelAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("PedestalsRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout << "Record \"PedestalsRcd"
                << "\" does not exist " << std::endl;
    }
    std::cout << "got context" << std::endl;
    auto const& myped = &context.getData(thePedestalToken_);
    std::cout << "Pedestals* " << myped << std::endl;
    for (std::vector<Pedestals::Item>::const_iterator it = myped->m_pedestals.begin(); it != myped->m_pedestals.end();
         ++it)
      std::cout << " mean: " << it->m_mean << " variance: " << it->m_variance;
    std::cout << std::endl;
  }
  DEFINE_FWK_MODULE(PedestalsByLabelAnalyzer);
}  // namespace edmtest
