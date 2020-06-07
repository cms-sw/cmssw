#include <iostream>
#include <fstream>
#include <stdexcept>

#include "CondTools/L1TriggerExt/interface/L1ConfigOnlineProdBaseExt.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsO2ORcd.h"

class L1TMuonOverlapParamsOnlineProd
    : public L1ConfigOnlineProdBaseExt<L1TMuonOverlapParamsO2ORcd, L1TMuonOverlapParams> {
private:
  bool transactionSafe;

public:
  std::unique_ptr<const L1TMuonOverlapParams> newObject(const std::string& objectKey,
                                                        const L1TMuonOverlapParamsO2ORcd& record) override;

  L1TMuonOverlapParamsOnlineProd(const edm::ParameterSet&);
  ~L1TMuonOverlapParamsOnlineProd(void) override {}
};

L1TMuonOverlapParamsOnlineProd::L1TMuonOverlapParamsOnlineProd(const edm::ParameterSet& iConfig)
    : L1ConfigOnlineProdBaseExt<L1TMuonOverlapParamsO2ORcd, L1TMuonOverlapParams>(iConfig) {
  m_setWhatProduced(iConfig);
  transactionSafe = iConfig.getParameter<bool>("transactionSafe");
}

std::unique_ptr<const L1TMuonOverlapParams> L1TMuonOverlapParamsOnlineProd::newObject(
    const std::string& objectKey, const L1TMuonOverlapParamsO2ORcd& record) {
  edm::LogError("L1-O2O") << "L1TMuonOverlapParams object with key " << objectKey << " not in ORCON!";

  if (transactionSafe)
    throw std::runtime_error(
        "SummaryForFunctionManager: OMTF  | Faulty  | You are never supposed to get OMTF online producer running!");

  auto retval = std::make_unique<const L1TMuonOverlapParams>();

  edm::LogError("L1-O2O: L1TMuonOverlapParamsOnlineProd")
      << "SummaryForFunctionManager: OMTF  | Faulty  | You are never supposed to get OMTF online producer running; "
         "returning empty L1TMuonOverlapParams";
  return retval;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapParamsOnlineProd);
