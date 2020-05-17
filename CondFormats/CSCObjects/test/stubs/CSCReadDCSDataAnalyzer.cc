#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/CSCObjects/interface/CSCDQM_DCSData.h"
#include "CondFormats/DataRecord/interface/CSCDCSDataRcd.h"

using namespace std;

namespace edmtest {

  class CSCReadDCSDataAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCReadDCSDataAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCReadDCSDataAnalyzer(int i) {}
    ~CSCReadDCSDataAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCReadDCSDataAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;

    std::cout << "+===================+" << std::endl;
    std::cout << "| CSCReadDCSDataAnalyzer    |" << std::endl;
    std::cout << "+===================+" << std::endl;

    std::cout << "run " << e.id().run() << std::endl;
    std::cout << "event " << e.id().event() << std::endl;

    edm::ESHandle<cscdqm::DCSData> hcrate;
    context.get<CSCDCSDataRcd>().get(hcrate);
    const cscdqm::DCSData* data = hcrate.product();

    std::cout << "Temp mode = " << data->temp_mode << std::endl;
    for (const auto& temp_mea : data->temp_meas)
      std::cout << temp_mea << std::endl;

    std::cout << "HV V mode = " << data->hvv_mode << std::endl;
    for (const auto& hvv_mea : data->hvv_meas)
      std::cout << hvv_mea << std::endl;

    std::cout << "LV V mode = " << data->lvv_mode << std::endl;
    for (const auto& lvv_mea : data->lvv_meas)
      std::cout << lvv_mea << std::endl;

    std::cout << "LV I mode = " << data->lvi_mode << std::endl;
    for (const auto& lvi_mea : data->lvi_meas)
      std::cout << lvi_mea << std::endl;

    std::cout << "+==========================+" << std::endl;
    std::cout << "| End of CSCReadDCSDataAnalyzer |" << std::endl;
    std::cout << "+==========================+" << std::endl;
  }

  DEFINE_FWK_MODULE(CSCReadDCSDataAnalyzer);

}  // namespace edmtest
