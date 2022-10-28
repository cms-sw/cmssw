#include <iostream>
#include <fstream>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapFwVersion.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionO2ORcd.h"
#include "L1Trigger/L1TCommon/interface/TriggerSystem.h"
#include "L1Trigger/L1TCommon/interface/XmlConfigParser.h"
#include "OnlineDBqueryHelper.h"

#include "xercesc/util/PlatformUtils.hpp"
using namespace XERCES_CPP_NAMESPACE;

class L1TMuonOverlapFwVersionOnlineProd
    : public L1ConfigOnlineProdBaseExt<L1TMuonOverlapFwVersionO2ORcd, L1TMuonOverlapFwVersion> {
private:
  const bool transactionSafe;
  const edm::ESGetToken<L1TMuonOverlapFwVersion, L1TMuonOverlapFwVersionRcd> baseSettings_token;

public:
  std::unique_ptr<const L1TMuonOverlapFwVersion> newObject(const std::string& objectKey,
                                                           const L1TMuonOverlapFwVersionO2ORcd& record) override;

  L1TMuonOverlapFwVersionOnlineProd(const edm::ParameterSet&);
  ~L1TMuonOverlapFwVersionOnlineProd(void) override = default;
};

void replaceAll(std::string& str, const std::string& from, const std::string& to) {
  if (from.empty())
    return;
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}

void removeAll(std::string& str, const std::string& from, const std::string& to) {
  if (from.empty())
    return;
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    size_t end_pos = str.find(to) + to.length();
    int length = end_pos - start_pos;
    str.replace(start_pos, length, "");
  }
}

L1TMuonOverlapFwVersionOnlineProd::L1TMuonOverlapFwVersionOnlineProd(const edm::ParameterSet& iConfig)
    : L1ConfigOnlineProdBaseExt<L1TMuonOverlapFwVersionO2ORcd, L1TMuonOverlapFwVersion>(iConfig),
      transactionSafe(iConfig.getParameter<bool>("transactionSafe")),
      baseSettings_token(wrappedSetWhatProduced(iConfig).consumes()) {}

std::unique_ptr<const L1TMuonOverlapFwVersion> L1TMuonOverlapFwVersionOnlineProd::newObject(
    const std::string& objectKey, const L1TMuonOverlapFwVersionO2ORcd& record) {
  const L1TMuonOverlapFwVersionRcd& baseRcd = record.template getRecord<L1TMuonOverlapFwVersionRcd>();
  auto const& baseSettings = baseRcd.get(baseSettings_token);

  if (objectKey.empty()) {
    edm::LogError("L1-O2O: L1TMuonOverlapFwVersionOnlineProd")
        << "Key is empty, returning empty L1TMuonOverlapFwVersion";
    if (transactionSafe)
      throw std::runtime_error("SummaryForFunctionManager: OMTF  | Faulty  | Empty objectKey");
    else {
      edm::LogError("L1-O2O: L1TMuonOverlapFwVersionOnlineProd")
          << "returning unmodified prototype of L1TMuonOverlapFwVersion";
      return std::make_unique<const L1TMuonOverlapFwVersion>(baseSettings);
    }
  }

  edm::LogInfo("L1-O2O: L1TMuonOverlapFwVersionOnlineProd")
      << "Producing L1TMuonOverlapFwVersion for key = " << objectKey;

  std::string payload, hw_fake;
  std::string algoV_string, layersV_string, patternsV_string, synthDate;

  try {
    payload = l1t::OnlineDBqueryHelper::fetch({"CONF"}, "OMTF_CLOBS", objectKey, m_omdsReader)["CONF"];

  } catch (std::runtime_error& e) {
    edm::LogError("L1-O2O: L1TMuonOverlapFwVersionOnlineProd") << e.what();
    if (transactionSafe)
      throw std::runtime_error(std::string("SummaryForFunctionManager: OMTF  | Faulty  | ") + e.what());
    else {
      edm::LogError("L1-O2O: L1TMuonOverlapFwVersionOnlineProd")
          << "returning unmodified prototype of L1TMuonOverlapFwVersion";
      return std::make_unique<const L1TMuonOverlapFwVersion>(baseSettings);
    }
  }
  // for debugging dump the configs to local files
  {
    std::ofstream output(std::string("/tmp/").append(objectKey.substr(0, objectKey.find('/'))).append(".xml"));
    output << objectKey;
    output.close();
  }

  // finally push all payloads to the XML parser and construct the TrigSystem object
  l1t::XmlConfigParser xmlRdr;
  l1t::TriggerSystem parsedXMLs;

  try {
    // no need to read all the HW settings, just define a dummy processor
    hw_fake = "<system id=\"OMTF\">  </system>";
    xmlRdr.readDOMFromString(hw_fake);
    parsedXMLs.addProcessor("processors", "processors", "all_crates", "all_slots");
    xmlRdr.readRootElement(parsedXMLs);

    // INFRA payload needs some editing to be suitable for the standard XML parser
    replaceAll(payload, "infra", "algo");
    removeAll(payload, "<context id=\"daq", "</context>");
    removeAll(payload, "<context id=\"OMTF", "</context>");
    xmlRdr.readDOMFromString(payload);
    xmlRdr.readRootElement(parsedXMLs);

    parsedXMLs.setConfigured();

  } catch (std::runtime_error& e) {
    edm::LogError("L1-O2O: L1TMuonOverlapFwVersionOnlineProd") << e.what();
    if (transactionSafe)
      throw std::runtime_error(std::string("SummaryForFunctionManager: OMTF  | Faulty at parsing XML  | ") + e.what());
    else {
      edm::LogError("L1-O2O: L1TMuonOverlapFwVersionOnlineProd")
          << "returning unmodified prototype of L1TMuonOverlapFwVersion";
      return std::make_unique<const L1TMuonOverlapFwVersion>(baseSettings);
    }
  }

  std::map<std::string, l1t::Parameter> conf = parsedXMLs.getParameters("processors");
  algoV_string = conf["algorithmVer"].getValueAsStr();
  layersV_string = conf["layersVer"].getValueAsStr();
  patternsV_string = conf["patternsVer"].getValueAsStr();
  synthDate = conf["synthDate"].getValueAsStr();

  unsigned algoV, layersV, patternsV;
  std::stringstream ssalgoV, sslayersV, sspatternsV;
  ssalgoV << std::hex << algoV_string.c_str();
  ssalgoV >> algoV;
  sslayersV << std::hex << layersV_string.c_str();
  sslayersV >> layersV;
  sspatternsV << std::hex << patternsV_string.c_str();
  sspatternsV >> patternsV;
  auto retval = std::make_unique<const L1TMuonOverlapFwVersion>(algoV, layersV, patternsV, synthDate);

  edm::LogInfo("L1-O2O: L1TMuonOverlapFwVersionOnlineProd")
      << "SummaryForFunctionManager: OMTF  | OK      | All looks good";
  return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapFwVersionOnlineProd);
