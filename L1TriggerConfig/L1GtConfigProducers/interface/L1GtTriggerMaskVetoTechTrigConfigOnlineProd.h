#ifndef L1GtConfigProducers_L1GtTriggerMaskVetoTechTrigConfigOnlineProd_h
#define L1GtConfigProducers_L1GtTriggerMaskVetoTechTrigConfigOnlineProd_h

/**
 * \class L1GtTriggerMaskVetoTechTrigConfigOnlineProd
 *
 *
 * Description: online producer for L1GtTriggerMaskVetoTechTrigRcd.
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

#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"

// forward declarations

// class declaration
class L1GtTriggerMaskVetoTechTrigConfigOnlineProd
    : public L1ConfigOnlineProdBase<L1GtTriggerMaskVetoTechTrigRcd, L1GtTriggerMask> {
public:
  /// constructor
  L1GtTriggerMaskVetoTechTrigConfigOnlineProd(const edm::ParameterSet&);

  /// destructor
  ~L1GtTriggerMaskVetoTechTrigConfigOnlineProd() override;

  /// public methods
  std::unique_ptr<L1GtTriggerMask> newObject(const std::string& objectKey) override;

private:
  /// partition number
  int m_partitionNumber;
};

#endif
