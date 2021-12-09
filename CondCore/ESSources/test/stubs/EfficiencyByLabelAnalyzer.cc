/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include <typeinfo>

#include "CondFormats/Calibration/interface/Efficiency.h"
#include "CondFormats/DataRecord/interface/ExEfficiency.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;

namespace edmtest {
  class EfficiencyByLabelAnalyzer : public edm::EDAnalyzer {
  public:
    explicit EfficiencyByLabelAnalyzer(edm::ParameterSet const& p)
        : theEffToken1_(esConsumes()), theEffToken2_(esConsumes()) {
      std::cout << "EfficiencyByLabelAnalyzer" << std::endl;
    }
    explicit EfficiencyByLabelAnalyzer(int i) { std::cout << "EfficiencyByLabelAnalyzer " << i << std::endl; }
    virtual ~EfficiencyByLabelAnalyzer() { std::cout << "~EfficiencyByLabelAnalyzer " << std::endl; }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    const edm::ESGetToken<condex::Efficiency, ExEfficiencyRcd> theEffToken1_, theEffToken2_;
  };

  void EfficiencyByLabelAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("ExEfficiencyRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout << "Record \"ExEfficiencyRcd"
                << "\" does not exist " << std::endl;
    }
    std::cout << "got context" << std::endl;
    {
      condex::Efficiency const& eff = context.getData(theEffToken2_);
      std::cout << "Efficiency*, type (2) " << (void*)(&eff) << " " << typeid(eff).name() << std::endl;
    }
    condex::Efficiency const& eff = context.getData(theEffToken1_);
    std::cout << "Efficiency*, type " << (void*)(&eff) << " " << typeid(eff).name() << std::endl;
    for (float pt = 0; pt < 10; pt += 2) {
      std::cout << "\npt=" << pt << "    :";
      for (float eta = -3; eta < 3; eta += 1)
        std::cout << eff(pt, eta) << " ";
    }
    std::cout << std::endl;
  }

  DEFINE_FWK_MODULE(EfficiencyByLabelAnalyzer);
}  // namespace edmtest
