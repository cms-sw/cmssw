#ifndef L1GtConfigProducers_L1GtPrescaleFactorsTechTrigConfigOnlineProd_h
#define L1GtConfigProducers_L1GtPrescaleFactorsTechTrigConfigOnlineProd_h

/**
 * \class L1GtPrescaleFactorsTechTrigConfigOnlineProd
 *
 *
 * Description: online producer for L1GtPrescaleFactorsTechTrigRcd.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// system include files
#include <memory>
#include <string>

// user include files
//   base class
#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

// forward declarations

// class declaration
class L1GtPrescaleFactorsTechTrigConfigOnlineProd
    : public L1ConfigOnlineProdBase<L1GtPrescaleFactorsTechTrigRcd, L1GtPrescaleFactors> {
public:
  /// constructor
  L1GtPrescaleFactorsTechTrigConfigOnlineProd(const edm::ParameterSet&);

  /// destructor
  ~L1GtPrescaleFactorsTechTrigConfigOnlineProd() override;

  /// public methods
  std::unique_ptr<L1GtPrescaleFactors> newObject(const std::string& objectKey) override;

private:
  bool m_isDebugEnabled;
};

#endif
