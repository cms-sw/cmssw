#include <iostream>
#include <fstream>
#include <strstream>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsO2ORcd.h"
#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParamsHelper.h"
#include "L1Trigger/L1TMuon/interface/L1TMuonGlobalParams_PUBLIC.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "OnlineDBqueryHelper.h"

class L1TMuonGlobalParamsOnlineProd : public L1ConfigOnlineProdBaseExt<L1TMuonGlobalParamsO2ORcd, L1TMuonGlobalParams> {
private:
  bool transactionSafe;
  edm::ESGetToken<L1TMuonGlobalParams,L1TMuonGlobalParamsRcd> baseSettings_token;

public:
  std::unique_ptr<const L1TMuonGlobalParams> newObject(const std::string &objectKey,
                                                       const L1TMuonGlobalParamsO2ORcd &record) override;

  L1TMuonGlobalParamsOnlineProd(const edm::ParameterSet &);
  ~L1TMuonGlobalParamsOnlineProd(void) override {}
};

L1TMuonGlobalParamsOnlineProd::L1TMuonGlobalParamsOnlineProd(const edm::ParameterSet &iConfig)
    : L1ConfigOnlineProdBaseExt<L1TMuonGlobalParamsO2ORcd, L1TMuonGlobalParams>(iConfig) {
  m_setWhatProduced(iConfig)
    .setConsumes(baseSettings_token);
  transactionSafe = iConfig.getParameter<bool>("transactionSafe");
}

std::unique_ptr<const L1TMuonGlobalParams> L1TMuonGlobalParamsOnlineProd::newObject(
    const std::string &objectKey, const L1TMuonGlobalParamsO2ORcd &record) {
  const L1TMuonGlobalParamsRcd &baseRcd = record.template getRecord<L1TMuonGlobalParamsRcd>();
  auto const baseSettings = baseRcd.get(baseSettings_token);

  if (objectKey.empty()) {
    edm::LogError("L1-O2O: L1TMuonGlobalParamsOnlineProd") << "Key is empty";
    if (transactionSafe)
      throw std::runtime_error("SummaryForFunctionManager: uGMT  | Faulty  | Empty objectKey");
    else {
      edm::LogError("L1-O2O: L1TMuonGlobalParams") << "returning unmodified prototype of L1TMuonGlobalParams";
      return std::make_unique<const L1TMuonGlobalParams>(baseSettings);
    }
  }

  std::string tscKey = objectKey.substr(0, objectKey.find(":"));
  std::string rsKey = objectKey.substr(objectKey.find(":") + 1, std::string::npos);

  edm::LogInfo("L1-O2O: L1TMuonGlobalParamsOnlineProd")
      << "Producing L1TMuonGlobalParams with TSC key =" << tscKey << " and RS key = " << rsKey;

  std::string algo_key, hw_key;
  std::string hw_payload;
  std::map<std::string, std::string> rs_payloads, algo_payloads;
  try {
    std::map<std::string, std::string> keys =
        l1t::OnlineDBqueryHelper::fetch({"ALGO", "HW"}, "UGMT_KEYS", tscKey, m_omdsReader);
    algo_key = keys["ALGO"];
    hw_key = keys["HW"];

    hw_payload = l1t::OnlineDBqueryHelper::fetch({"CONF"}, "UGMT_CLOBS", hw_key, m_omdsReader)["CONF"];

    std::map<std::string, std::string> rsKeys =
        l1t::OnlineDBqueryHelper::fetch({"MP7", "MP7_MONI", "AMC13_MONI"}, "UGMT_RS_KEYS", rsKey, m_omdsReader);

    std::map<std::string, std::string> algoKeys =
        l1t::OnlineDBqueryHelper::fetch({"MP7", "LUTS"}, "UGMT_ALGO_KEYS", algo_key, m_omdsReader);

    for (auto &key : rsKeys)
      rs_payloads[key.second] =
          l1t::OnlineDBqueryHelper::fetch({"CONF"}, "UGMT_CLOBS", key.second, m_omdsReader)["CONF"];

    for (auto &key : algoKeys)
      algo_payloads[key.second] =
          l1t::OnlineDBqueryHelper::fetch({"CONF"}, "UGMT_CLOBS", key.second, m_omdsReader)["CONF"];
  } catch (std::runtime_error &e) {
    edm::LogError("L1-O2O: L1TMuonGlobalParamsOnlineProd") << e.what();
    if (transactionSafe)
      throw std::runtime_error("SummaryForFunctionManager: uGMT  | Faulty  | Broken key");
    else {
      edm::LogError("L1-O2O: L1TMuonGlobalParamsOnlineProd") << "returning unmodified prototype of L1TMuonGlobalParams";
      return std::make_unique<const L1TMuonGlobalParams>(baseSettings);
    }
  }

  // for debugging dump the configs to local files
  {
    std::ofstream output(std::string("/tmp/").append(hw_key.substr(0, hw_key.find("/"))).append(".xml"));
    output << hw_payload;
    output.close();
  }
  for (auto &conf : rs_payloads) {
    std::ofstream output(std::string("/tmp/").append(conf.first.substr(0, conf.first.find("/"))).append(".xml"));
    output << conf.second;
    output.close();
  }
  for (auto &conf : algo_payloads) {
    std::ofstream output(std::string("/tmp/").append(conf.first.substr(0, conf.first.find("/"))).append(".xml"));
    output << conf.second;
    output.close();
  }

  // finally, push all payloads to the XML parser and construct the TrigSystem objects with each of those
  l1t::XmlConfigParser xmlRdr;
  l1t::TriggerSystem trgSys;

  try {
    // HW settings should always go first
    xmlRdr.readDOMFromString(hw_payload);
    xmlRdr.readRootElement(trgSys);

    // now let's parse ALGO and then RS settings
    for (auto &conf : algo_payloads) {
      xmlRdr.readDOMFromString(conf.second);
      xmlRdr.readRootElement(trgSys);
    }
    for (auto &conf : rs_payloads) {
      xmlRdr.readDOMFromString(conf.second);
      xmlRdr.readRootElement(trgSys);
    }
    trgSys.setConfigured();
  } catch (std::runtime_error &e) {
    edm::LogError("L1-O2O: L1TMuonGlobalParamsOnlineProd") << e.what();
    if (transactionSafe)
      throw std::runtime_error("SummaryForFunctionManager: uGMT  | Faulty  | Cannot parse XMLs");
    else {
      edm::LogError("L1-O2O: L1TMuonGlobalParamsOnlineProd") << "returning unmodified prototype of L1TMuonGlobalParams";
      return std::make_unique<const L1TMuonGlobalParams>(baseSettings);
    }
  }

  L1TMuonGlobalParamsHelper m_params_helper(baseSettings);
  try {
    m_params_helper.loadFromOnline(trgSys);
  } catch (std::runtime_error &e) {
    edm::LogError("L1-O2O: L1TMuonGlobalParamsOnlineProd") << e.what();
    if (transactionSafe)
      throw std::runtime_error("SummaryForFunctionManager: uGMT  | Faulty  | Cannot run helper");
    else {
      edm::LogError("L1-O2O: L1TMuonGlobalParamsOnlineProd") << "returning unmodified prototype of L1TMuonGlobalParams";
      return std::make_unique<const L1TMuonGlobalParams>(baseSettings);
    }
  }

  auto retval = std::make_unique<const L1TMuonGlobalParams>(cast_to_L1TMuonGlobalParams(m_params_helper));

  edm::LogInfo("L1-O2O: L1TMuonGlobalParamsOnlineProd")
      << "SummaryForFunctionManager: uGMT  | OK      | All looks good";
  return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonGlobalParamsOnlineProd);
