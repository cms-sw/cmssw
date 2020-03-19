#ifndef L1GCTEMULATOR_H
#define L1GCTEMULATOR_H

/**\class L1GctEmulator L1GctEmulator.h src/L1Trigger/GlobalCaloTrigger/src/L1GctEmulator.h

 Description:  Framework module that runs the GCT bit-level emulator

 Implementation:
       An EDProducer that contains an instance of L1GlobalCaloTrigger.
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu May 18 15:04:56 CEST 2006
//
//

// system includes

// EDM includes
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

class L1GctEmulator : public edm::stream::EDProducer<> {
public:
  /// typedefs
  typedef L1GlobalCaloTrigger::lutPtr lutPtr;
  typedef L1GlobalCaloTrigger::lutPtrVector lutPtrVector;

  /// constructor
  explicit L1GctEmulator(const edm::ParameterSet& ps);

private:
  void produce(edm::Event& e, const edm::EventSetup& c) override;

  int configureGct(const edm::EventSetup& c);

  // input label
  std::string m_inputLabel;
  edm::EDGetTokenT<L1CaloEmCollection> m_emToken;
  edm::EDGetTokenT<L1CaloRegionCollection> m_regionToken;

  // pointer to the actual emulator
  std::unique_ptr<L1GlobalCaloTrigger> m_gct;

  // pointers to the jet Et LUTs
  lutPtrVector m_jetEtCalibLuts;

  // data output switch
  const bool m_writeInternalData;

  // untracked parameters
  const bool m_verbose;

  // label for conditions
  const std::string m_conditionsLabel;

  // tracked parameters
};

#endif
