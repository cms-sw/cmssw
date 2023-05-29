#ifndef GlobalTriggerAnalyzer_L1GtBeamModeFilter_h
#define GlobalTriggerAnalyzer_L1GtBeamModeFilter_h

/**
 * \class L1GtBeamModeFilter
 *
 *
 * Description: filter according to the beam mode using the BST information received by L1 GT.
 *
 * Implementation:
 *    This class is an EDFilter. It implements:
 *      - filter according to the beam mode using the BST information received by L1 GT
 *      - the code for a given mode is the code given in the BST message document
 *        (LHC-BOB-ES-0001-20-00.pdf or newer version)
 *      - it requires as input the unpacked L1GlobalTriggerEvmReadoutRecord or
 *        the ConditionsInEdm product
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// system include files
#include <vector>

// user include files

//   base class
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// forward declarations

// class declaration
class L1GtBeamModeFilter : public edm::global::EDFilter<> {
public:
  /// constructor
  explicit L1GtBeamModeFilter(const edm::ParameterSet&);

  /// destructor
  ~L1GtBeamModeFilter() override;

  /// filter the event
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  /// input tag for ConditionInEdm products
  edm::InputTag m_condInEdmInputTag;

  /// InputTag for the L1 Global Trigger EVM readout record
  edm::InputTag m_l1GtEvmReadoutRecordTag;

  /// vector of beam modes (coded as integer numbers)
  std::vector<unsigned int> m_allowedBeamMode;

  /// return the inverted result
  bool m_invertResult;

  /// cache edm::isDebugEnabled()
  bool m_isDebugEnabled;
};

#endif  // GlobalTriggerAnalyzer_L1GtBeamModeFilter_h
