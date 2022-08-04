#ifndef L1GtConfigProducers_L1GctConfigDump_h
#define L1GtConfigProducers_L1GctConfigDump_h

/**
 * \class L1GctConfigDump
 * 
 * 
 * Description: test analyzer for L1 GCT parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Jim Brooke
 * 
 *
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class L1GctJetFinderParams;
class L1GctChannelMask;
class L1CaloEtScale;
class L1GctJetFinderParamsRcd;
class L1GctChannelMaskRcd;
class L1JetEtScaleRcd;
class L1HtMissScaleRcd;
class L1HfRingEtScaleRcd;

// class declaration
class L1GctConfigDump : public edm::one::EDAnalyzer<> {
public:
  // constructor
  explicit L1GctConfigDump(const edm::ParameterSet&);

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<L1GctJetFinderParams, L1GctJetFinderParamsRcd> m_jfParamsToken;
  const edm::ESGetToken<L1GctChannelMask, L1GctChannelMaskRcd> m_chanMaskToken;
  const edm::ESGetToken<L1CaloEtScale, L1JetEtScaleRcd> m_jetScaleToken;
  const edm::ESGetToken<L1CaloEtScale, L1HtMissScaleRcd> m_htmScaleToken;
  const edm::ESGetToken<L1CaloEtScale, L1HfRingEtScaleRcd> m_hfRingScaleToken;
};

#endif
