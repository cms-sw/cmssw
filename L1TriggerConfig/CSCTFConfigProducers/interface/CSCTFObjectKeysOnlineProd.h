#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"

class CSCTFObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
public:
  CSCTFObjectKeysOnlineProd(const edm::ParameterSet& iConfig);
  ~CSCTFObjectKeysOnlineProd() override {}

  void fillObjectKeys(FillType) override;

private:
  bool m_enableConfiguration;
  bool m_enablePtLut;
};
