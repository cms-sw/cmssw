#include <iostream>
#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "OnlineDBqueryHelper.h"

class L1TMuonOverlapObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBaseExt {
private:
  bool transactionSafe;

public:
  void fillObjectKeys(L1TriggerKeyExt* pL1TriggerKey) override;

  L1TMuonOverlapObjectKeysOnlineProd(const edm::ParameterSet&);
  ~L1TMuonOverlapObjectKeysOnlineProd(void) override = default;
};

L1TMuonOverlapObjectKeysOnlineProd::L1TMuonOverlapObjectKeysOnlineProd(const edm::ParameterSet& iConfig)
    : L1ObjectKeysOnlineProdBaseExt(iConfig) {
  transactionSafe = iConfig.getParameter<bool>("transactionSafe");
}

void L1TMuonOverlapObjectKeysOnlineProd::fillObjectKeys(L1TriggerKeyExt* pL1TriggerKey) {
  std::string OMTFKey = pL1TriggerKey->subsystemKey(L1TriggerKeyExt::kOMTF);

  std::string tscKey = OMTFKey.substr(0, OMTFKey.find(':'));
  std::string algo_key, infra_key;

  // L1TMuonOverlapFwVersion and L1TMuonOverlapParams keys to be found from INFRA and ALGO, respectively

  try {
    std::map<std::string, std::string> keys =
        l1t::OnlineDBqueryHelper::fetch({"ALGO", "INFRA"}, "OMTF_KEYS", tscKey, m_omdsReader);
    algo_key = keys["ALGO"];
    infra_key = keys["INFRA"];

  } catch (std::runtime_error& e) {
    edm::LogError("L1-O2O L1TMuonOverlapObjectKeysOnlineProd") << "Cannot get OMTF_KEYS ";

    if (transactionSafe)
      throw std::runtime_error("SummaryForFunctionManager: OMTF  | Faulty  | Broken key");
    else {
      edm::LogError("L1-O2O: L1TMuonOverlapObjectKeysOnlineProd")
          << "forcing L1TMuonOverlapFwVersion key to be = 'OMTF_INFRA_EMPTY' with baseline settings";
      pL1TriggerKey->add("L1TMuonOverlapFwVersionO2ORcd", "L1TMuonOverlapFwVersion", "OMTF_INFRA_EMPTY");
      edm::LogError("L1-O2O: L1TMuonOverlapObjectKeysOnlineProd")
          << "forcing L1TMuonOverlapParams key to be = 'OMTF_ALGO_EMPTY' (known to exist)";
      pL1TriggerKey->add("L1TMuonOverlapParamsO2ORcd", "L1TMuonOverlapParams", "OMTF_ALGO_EMPTY");
      return;
    }
  }

  pL1TriggerKey->add("L1TMuonOverlapFwVersionO2ORcd", "L1TMuonOverlapFwVersion", infra_key);

  pL1TriggerKey->add("L1TMuonOverlapParamsO2ORcd", "L1TMuonOverlapParams", algo_key);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapObjectKeysOnlineProd);
