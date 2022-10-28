#include "L1TriggerConfig/GctConfigProducers/interface/L1GctConfigDump.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"

#include <iomanip>

L1GctConfigDump::L1GctConfigDump(const edm::ParameterSet& pSet)
    : m_jfParamsToken{esConsumes()},
      m_chanMaskToken{esConsumes()},
      m_jetScaleToken{esConsumes()},
      m_htmScaleToken{esConsumes()},
      m_hfRingScaleToken{esConsumes()} {
  // empty
}

void L1GctConfigDump::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get records

  edm::ESHandle<L1GctJetFinderParams> jfParams = iSetup.getHandle(m_jfParamsToken);

  edm::ESHandle<L1GctChannelMask> chanMask = iSetup.getHandle(m_chanMaskToken);

  edm::ESHandle<L1CaloEtScale> jetScale = iSetup.getHandle(m_jetScaleToken);

  edm::ESHandle<L1CaloEtScale> htmScale = iSetup.getHandle(m_htmScaleToken);

  edm::ESHandle<L1CaloEtScale> hfRingScale = iSetup.getHandle(m_hfRingScaleToken);

  edm::LogInfo("L1GctConfigDump") << (*jfParams) << std::endl;
  edm::LogInfo("L1GctConfigDump") << (*chanMask) << std::endl;
  edm::LogInfo("L1GctConfigDump") << "GCT jet Et scale : " << std::endl << (*jetScale) << std::endl;
  edm::LogInfo("L1GctConfigDump") << "GCT HtMiss scale : " << std::endl << (*htmScale) << std::endl;
  edm::LogInfo("L1GctConfigDump") << "GCT HF ring scale : " << std::endl << (*hfRingScale) << std::endl;
}

DEFINE_FWK_MODULE(L1GctConfigDump);
