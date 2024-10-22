#include <iostream>
#include <fstream>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsO2ORcd.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelParamsHelper.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "OnlineDBqueryHelper.h"

#include "xercesc/util/PlatformUtils.hpp"
using namespace XERCES_CPP_NAMESPACE;

class L1TMuonBarrelParamsOnlineProd : public L1ConfigOnlineProdBaseExt<L1TMuonBarrelParamsO2ORcd, L1TMuonBarrelParams> {
private:
  const bool transactionSafe;
  const edm::ESGetToken<L1TMuonBarrelParams, L1TMuonBarrelParamsRcd> baseSettings_token;

public:
  std::unique_ptr<const L1TMuonBarrelParams> newObject(const std::string& objectKey,
                                                       const L1TMuonBarrelParamsO2ORcd& record) override;

  L1TMuonBarrelParamsOnlineProd(const edm::ParameterSet&);
  ~L1TMuonBarrelParamsOnlineProd(void) override {}
};

L1TMuonBarrelParamsOnlineProd::L1TMuonBarrelParamsOnlineProd(const edm::ParameterSet& iConfig)
    : L1ConfigOnlineProdBaseExt<L1TMuonBarrelParamsO2ORcd, L1TMuonBarrelParams>(iConfig),
      transactionSafe(iConfig.getParameter<bool>("transactionSafe")),
      baseSettings_token(wrappedSetWhatProduced(iConfig).consumes()) {}

std::unique_ptr<const L1TMuonBarrelParams> L1TMuonBarrelParamsOnlineProd::newObject(
    const std::string& objectKey, const L1TMuonBarrelParamsO2ORcd& record) {
  const L1TMuonBarrelParamsRcd& baseRcd = record.template getRecord<L1TMuonBarrelParamsRcd>();
  auto const& baseSettings = baseRcd.get(baseSettings_token);

  if (objectKey.empty()) {
    edm::LogError("L1-O2O: L1TMuonBarrelParamsOnlineProd") << "Key is empty, returning empty L1TMuonBarrelParams";
    if (transactionSafe)
      throw std::runtime_error("SummaryForFunctionManager: BMTF  | Faulty  | Empty objectKey");
    else {
      edm::LogError("L1-O2O: L1TMuonBarrelParamsOnlineProd") << "returning unmodified prototype of L1TMuonBarrelParams";
      return std::make_unique<const L1TMuonBarrelParams>(baseSettings);
    }
  }

  std::string tscKey = objectKey.substr(0, objectKey.find(':'));
  std::string rsKey = objectKey.substr(objectKey.find(':') + 1, std::string::npos);

  edm::LogInfo("L1-O2O: L1TMuonBarrelParamsOnlineProd")
      << "Producing L1TMuonBarrelParams with TSC key = " << tscKey << " and RS key = " << rsKey;

  std::string algo_key, hw_key;
  std::string mp7_key, amc13_key;
  std::string hw_payload, algo_payload, mp7_payload, amc13_payload;
  try {
    std::map<std::string, std::string> keys =
        l1t::OnlineDBqueryHelper::fetch({"ALGO", "HW"}, "BMTF_KEYS", tscKey, m_omdsReader);
    algo_key = keys["ALGO"];
    hw_key = keys["HW"];

    hw_payload = l1t::OnlineDBqueryHelper::fetch({"CONF"}, "BMTF_CLOBS", hw_key, m_omdsReader)["CONF"];

    algo_payload = l1t::OnlineDBqueryHelper::fetch({"CONF"}, "BMTF_CLOBS", algo_key, m_omdsReader)["CONF"];

    std::map<std::string, std::string> rsKeys =
        l1t::OnlineDBqueryHelper::fetch({"MP7", "AMC13"}, "BMTF_RS_KEYS", rsKey, m_omdsReader);
    mp7_key = rsKeys["MP7"];
    amc13_key = rsKeys["AMC13"];

    mp7_payload = l1t::OnlineDBqueryHelper::fetch({"CONF"}, "BMTF_CLOBS", mp7_key, m_omdsReader)["CONF"];
    amc13_payload = l1t::OnlineDBqueryHelper::fetch({"CONF"}, "BMTF_CLOBS", amc13_key, m_omdsReader)["CONF"];

  } catch (std::runtime_error& e) {
    edm::LogError("L1-O2O: L1TMuonBarrelParamsOnlineProd") << e.what();
    if (transactionSafe)
      throw std::runtime_error(std::string("SummaryForFunctionManager: BMTF  | Faulty  | ") + e.what());
    else {
      edm::LogError("L1-O2O: L1TMuonBarrelParamsOnlineProd") << "returning unmodified prototype of L1TMuonBarrelParams";
      return std::make_unique<const L1TMuonBarrelParams>(baseSettings);
    }
  }

  // for debugging dump the configs to local files
  {
    std::ofstream output(std::string("/tmp/").append(hw_key.substr(0, hw_key.find('/'))).append(".xml"));
    output << hw_payload;
    output.close();
  }
  {
    std::ofstream output(std::string("/tmp/").append(algo_key.substr(0, algo_key.find('/'))).append(".xml"));
    output << algo_payload;
    output.close();
  }
  {
    std::ofstream output(std::string("/tmp/").append(mp7_key.substr(0, mp7_key.find('/'))).append(".xml"));
    output << mp7_payload;
    output.close();
  }
  {
    std::ofstream output(std::string("/tmp/").append(amc13_key.substr(0, amc13_key.find('/'))).append(".xml"));
    output << amc13_payload;
    output.close();
  }

  // finally, push all payloads to the XML parser and construct the TrigSystem objects with each of those
  l1t::XmlConfigParser xmlRdr;
  l1t::TriggerSystem parsedXMLs;
  try {
    // HW settings should always go first
    xmlRdr.readDOMFromString(hw_payload);
    xmlRdr.readRootElement(parsedXMLs);

    // now let's parse ALGO settings
    xmlRdr.readDOMFromString(algo_payload);
    xmlRdr.readRootElement(parsedXMLs);

    // remaining RS settings
    xmlRdr.readDOMFromString(mp7_payload);
    xmlRdr.readRootElement(parsedXMLs);

    xmlRdr.readDOMFromString(amc13_payload);
    xmlRdr.readRootElement(parsedXMLs);
    parsedXMLs.setConfigured();

  } catch (std::runtime_error& e) {
    edm::LogError("L1-O2O: L1TMuonBarrelParamsOnlineProd") << e.what();
    if (transactionSafe)
      throw std::runtime_error(std::string("SummaryForFunctionManager: BMTF  | Faulty  | ") + e.what());
    else {
      edm::LogError("L1-O2O: L1TMuonBarrelParamsOnlineProd") << "returning unmodified prototype of L1TMuonBarrelParams";
      return std::make_unique<const L1TMuonBarrelParams>(baseSettings);
    }
  }

  L1TMuonBarrelParamsHelper m_params_helper(baseSettings);
  try {
    m_params_helper.configFromDB(parsedXMLs);
  } catch (std::runtime_error& e) {
    edm::LogError("L1-O2O: L1TMuonBarrelParamsOnlineProd") << e.what();
    if (transactionSafe)
      throw std::runtime_error(std::string("SummaryForFunctionManager: BMTF  | Faulty  | ") + e.what());
    else {
      edm::LogError("L1-O2O: L1TMuonBarrelParamsOnlineProd") << "returning unmodified prototype of L1TMuonBarrelParams";
      return std::make_unique<const L1TMuonBarrelParams>(baseSettings);
    }
  }

  auto retval = std::make_unique<const L1TMuonBarrelParams>(m_params_helper);

  edm::LogInfo("L1-O2O: L1TMuonBarrelParamsOnlineProd")
      << "SummaryForFunctionManager: BMTF  | OK      | All looks good";
  return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonBarrelParamsOnlineProd);
