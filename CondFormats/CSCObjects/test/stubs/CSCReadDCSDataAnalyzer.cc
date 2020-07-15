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
    for (unsigned int i = 0; i < data->temp_meas.size(); i++)
      std::cout << data->temp_meas.at(i) << std::endl;

    std::cout << "HV V mode = " << data->hvv_mode << std::endl;
    for (unsigned int i = 0; i < data->hvv_meas.size(); i++)
      std::cout << data->hvv_meas.at(i) << std::endl;

    std::cout << "LV V mode = " << data->lvv_mode << std::endl;
    for (unsigned int i = 0; i < data->lvv_meas.size(); i++)
      std::cout << data->lvv_meas.at(i) << std::endl;

    std::cout << "LV I mode = " << data->lvi_mode << std::endl;
    for (unsigned int i = 0; i < data->lvi_meas.size(); i++)
      std::cout << data->lvi_meas.at(i) << std::endl;

    std::cout << "+==========================+" << std::endl;
    std::cout << "| End of CSCReadDCSDataAnalyzer |" << std::endl;
    std::cout << "+==========================+" << std::endl;
  }

  DEFINE_FWK_MODULE(CSCReadDCSDataAnalyzer);

}  // namespace edmtest
