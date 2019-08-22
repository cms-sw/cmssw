#ifndef L1GtConfigProducers_L1GtParametersConfigOnlineProd_h
#define L1GtConfigProducers_L1GtParametersConfigOnlineProd_h

/**
 * \class L1GtParametersConfigOnlineProd
 *
 *
 * Description: online producer for L1GtParameters.
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

#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"

// forward declarations

// class declaration
class L1GtParametersConfigOnlineProd : public L1ConfigOnlineProdBase<L1GtParametersRcd, L1GtParameters> {
public:
  /// constructor
  L1GtParametersConfigOnlineProd(const edm::ParameterSet&);

  /// destructor
  ~L1GtParametersConfigOnlineProd() override;

  /// public methods
  std::unique_ptr<L1GtParameters> newObject(const std::string& objectKey) override;

private:
  ///
};

#endif
