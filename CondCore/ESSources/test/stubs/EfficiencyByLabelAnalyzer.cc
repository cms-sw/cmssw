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

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;

namespace edmtest {
  class EfficiencyByLabelAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit EfficiencyByLabelAnalyzer(edm::ParameterSet const& p)
        : theEffToken1_(esConsumes()), theEffToken2_(esConsumes()) {
      edm::LogPrint("EfficiencyByLabelAnalyzer") << "EfficiencyByLabelAnalyzer";
    }
    explicit EfficiencyByLabelAnalyzer(int i) {
      edm::LogPrint("EfficiencyByLabelAnalyzer") << "EfficiencyByLabelAnalyzer " << i;
    }
    virtual ~EfficiencyByLabelAnalyzer() {
      edm::LogPrint("EfficiencyByLabelAnalyzer") << "~EfficiencyByLabelAnalyzer ";
    }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    const edm::ESGetToken<condex::Efficiency, ExEfficiencyRcd> theEffToken1_, theEffToken2_;
  };

  void EfficiencyByLabelAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    edm::LogPrint("EfficiencyByLabelAnalyzer") << " I AM IN RUN NUMBER " << e.id().run();
    edm::LogPrint("EfficiencyByLabelAnalyzer") << " ---EVENT NUMBER " << e.id().event();
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("ExEfficiencyRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogPrint("EfficiencyByLabelAnalyzer") << "Record \"ExEfficiencyRcd\" does not exist";
    }
    edm::LogPrint("EfficiencyByLabelAnalyzer") << "got context";
    {
      condex::Efficiency const& eff = context.getData(theEffToken2_);
      edm::LogPrint("EfficiencyByLabelAnalyzer")
          << "Efficiency*, type (2) " << (void*)(&eff) << " " << typeid(eff).name();
    }
    condex::Efficiency const& eff = context.getData(theEffToken1_);
    edm::LogPrint("EfficiencyByLabelAnalyzer") << "Efficiency*, type " << (void*)(&eff) << " " << typeid(eff).name();
    for (float pt = 0; pt < 10; pt += 2) {
      edm::LogPrint("EfficiencyByLabelAnalyzer") << "\npt=" << pt << "    :";
      for (float eta = -3; eta < 3; eta += 1)
        edm::LogPrint("EfficiencyByLabelAnalyzer") << eff(pt, eta) << " ";
    }
    edm::LogPrint("EfficiencyByLabelAnalyzer");
  }

  DEFINE_FWK_MODULE(EfficiencyByLabelAnalyzer);
}  // namespace edmtest
