#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFAlignment.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFAlignmentRcd.h"

class CSCTFAlignmentOnlineProd : public L1ConfigOnlineProdBase<L1MuCSCTFAlignmentRcd, L1MuCSCTFAlignment> {
public:
  CSCTFAlignmentOnlineProd(const edm::ParameterSet& iConfig)
      : L1ConfigOnlineProdBase<L1MuCSCTFAlignmentRcd, L1MuCSCTFAlignment>(iConfig) {}
  ~CSCTFAlignmentOnlineProd() override {}
  std::unique_ptr<L1MuCSCTFAlignment> newObject(const std::string& objectKey) override;

private:
};
