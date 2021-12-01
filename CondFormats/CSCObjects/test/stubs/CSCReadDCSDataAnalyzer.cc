#include <string>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/CSCObjects/interface/CSCDQM_DCSData.h"
#include "CondFormats/DataRecord/interface/CSCDCSDataRcd.h"

using namespace std;

namespace edmtest {

  class CSCReadDCSDataAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit CSCReadDCSDataAnalyzer(edm::ParameterSet const& p) : token_{esConsumes()} {}
    ~CSCReadDCSDataAnalyzer() override {}
    void analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& c) const override;

  private:
    edm::ESGetToken<cscdqm::DCSData, CSCDCSDataRcd> token_;
  };

  void CSCReadDCSDataAnalyzer::analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& context) const {
    using namespace edm::eventsetup;

    edm::LogSystem log("DCSData");

    log << "+===================+" << std::endl;
    log << "| CSCReadDCSDataAnalyzer    |" << std::endl;
    log << "+===================+" << std::endl;

    log << "run " << e.id().run() << std::endl;
    log << "event " << e.id().event() << std::endl;

    const cscdqm::DCSData* data = &context.getData(token_);

    log << "Temp mode = " << data->temp_mode << std::endl;
    for (unsigned int i = 0; i < data->temp_meas.size(); i++)
      log << data->temp_meas.at(i) << std::endl;

    log << "HV V mode = " << data->hvv_mode << std::endl;
    for (unsigned int i = 0; i < data->hvv_meas.size(); i++)
      log << data->hvv_meas.at(i) << std::endl;

    log << "LV V mode = " << data->lvv_mode << std::endl;
    for (unsigned int i = 0; i < data->lvv_meas.size(); i++)
      log << data->lvv_meas.at(i) << std::endl;

    log << "LV I mode = " << data->lvi_mode << std::endl;
    for (unsigned int i = 0; i < data->lvi_meas.size(); i++)
      log << data->lvi_meas.at(i) << std::endl;

    log << "+==========================+" << std::endl;
    log << "| End of CSCReadDCSDataAnalyzer |" << std::endl;
    log << "+==========================+" << std::endl;
  }

  DEFINE_FWK_MODULE(CSCReadDCSDataAnalyzer);

}  // namespace edmtest
