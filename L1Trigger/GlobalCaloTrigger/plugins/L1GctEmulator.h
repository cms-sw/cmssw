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
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"
// Trigger configuration includes
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"

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

  //EventSetup Tokens
  edm::ESGetToken<L1GctJetFinderParams, L1GctJetFinderParamsRcd> m_jfParsToken;
  edm::ESGetToken<L1GctChannelMask, L1GctChannelMaskRcd> m_chanMaskToken;
  edm::ESGetToken<L1CaloEtScale, L1JetEtScaleRcd> m_etScaleToken;
  edm::ESGetToken<L1CaloEtScale, L1HtMissScaleRcd> m_htMissScaleToken;
  edm::ESGetToken<L1CaloEtScale, L1HfRingEtScaleRcd> m_hfRingEtScaleToken;

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
