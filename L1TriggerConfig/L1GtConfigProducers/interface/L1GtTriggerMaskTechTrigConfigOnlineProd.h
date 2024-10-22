#ifndef L1GtConfigProducers_L1GtTriggerMaskTechTrigConfigOnlineProd_h
#define L1GtConfigProducers_L1GtTriggerMaskTechTrigConfigOnlineProd_h

/**
 * \class L1GtTriggerMaskTechTrigConfigOnlineProd
 *
 *
 * Description: online producer for L1GtTriggerMaskTechTrigRcd.
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
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

// forward declarations

// class declaration
class L1GtTriggerMaskTechTrigConfigOnlineProd
    : public L1ConfigOnlineProdBase<L1GtTriggerMaskTechTrigRcd, L1GtTriggerMask> {
public:
  /// constructor
  L1GtTriggerMaskTechTrigConfigOnlineProd(const edm::ParameterSet&);

  /// destructor
  ~L1GtTriggerMaskTechTrigConfigOnlineProd() override;

  /// public methods
  std::unique_ptr<L1GtTriggerMask> newObject(const std::string& objectKey) override;

private:
  /// partition number
  int m_partitionNumber;
};

#endif
