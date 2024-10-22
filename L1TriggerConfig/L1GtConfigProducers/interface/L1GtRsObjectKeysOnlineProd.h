#ifndef L1GtConfigProducers_L1GtRsObjectKeysOnlineProd_h
#define L1GtConfigProducers_L1GtRsObjectKeysOnlineProd_h

/**
 * \class L1GtRsObjectKeysOnlineProd
 *
 *
 * Description: online producer for L1 GT record keys from RUN SETTINGS.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// system include files

// user include files
//   base class
#include "CondTools/L1Trigger/interface/L1ObjectKeysOnlineProdBase.h"

class L1GtRsObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
public:
  /// constructor
  L1GtRsObjectKeysOnlineProd(const edm::ParameterSet&);

  /// destructor
  ~L1GtRsObjectKeysOnlineProd() override;

  /// public methods
  void fillObjectKeys(FillType pL1TriggerKey) override;

private:
  /// keys for individual objects
  std::string keyL1GtPrescaleFactorsAlgoTrig(const std::string&);
  std::string keyL1GtPrescaleFactorsTechTrig(const std::string&);
  std::string keyL1GtTriggerMaskAlgoTrig(const std::string&);
  std::string keyL1GtTriggerMaskTechTrig(const std::string&);
  std::string keyL1GtTriggerMaskVetoTechTrig(const std::string&);

private:
  /// partition number
  int m_partitionNumber;

  /// enable key search for each record
  bool m_enableL1GtPrescaleFactorsAlgoTrig;
  bool m_enableL1GtPrescaleFactorsTechTrig;
  bool m_enableL1GtTriggerMaskAlgoTrig;
  bool m_enableL1GtTriggerMaskTechTrig;
  bool m_enableL1GtTriggerMaskVetoTechTrig;
};

#endif
