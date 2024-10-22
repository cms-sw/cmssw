#ifndef L1GtConfigProducers_L1GtTscObjectKeysOnlineProd_h
#define L1GtConfigProducers_L1GtTscObjectKeysOnlineProd_h

/**
 * \class L1GtTscObjectKeysOnlineProd
 *
 *
 * Description: online producer for L1 GT record keys starting from TSC key.
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

class L1GtTscObjectKeysOnlineProd : public L1ObjectKeysOnlineProdBase {
public:
  /// constructor
  L1GtTscObjectKeysOnlineProd(const edm::ParameterSet&);

  /// destructor
  ~L1GtTscObjectKeysOnlineProd() override;

  /// public methods
  void fillObjectKeys(FillType pL1TriggerKey) override;

private:
  /// keys for individual objects
  std::string keyL1GtParameters(const std::string& subsystemKey, const std::string& gtSchema);
  std::string keyL1GtTriggerMenu(const std::string& subsystemKey, const std::string& gtSchema);
  std::string keyL1GtPsbSetup(const std::string& subsystemKey, const std::string& gtSchema);

private:
  /// enable key search for each record
  bool m_enableL1GtParameters;
  bool m_enableL1GtTriggerMenu;
  bool m_enableL1GtPsbSetup;
};

#endif
