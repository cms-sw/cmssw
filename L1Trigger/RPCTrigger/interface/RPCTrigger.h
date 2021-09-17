#ifndef L1Trigger_RPCTrigger_h
#define L1Trigger_RPCTrigger_h

/** \class RPCTrigger
 *  \brief Implements RPC trigger emulation
 *
 *  \author Tomasz Fruboes
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"  // Handle to read geometry
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
// Configuration via eventsetup:
#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"

#include "CondFormats/DataRecord/interface/L1RPCConeBuilderRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"

#include "CondFormats/DataRecord/interface/L1RPCHwConfigRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"

#include "CondFormats/DataRecord/interface/L1RPCBxOrConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCBxOrConfig.h"

// L1RpcTrigger specific includes
#include "L1Trigger/RPCTrigger/interface/RPCConeBuilderFromES.h"

#include "L1Trigger/RPCTrigger/interface/RPCPacManager.h"

#include "L1Trigger/RPCTrigger/interface/RPCPacTrigger.h"
#include "L1Trigger/RPCTrigger/interface/RPCBasicTrigConfig.h"
#include "L1Trigger/RPCTrigger/interface/RPCPacData.h"
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
#include "L1Trigger/RPCTrigger/interface/RPCPacManager.h"
#include "CondFormats/DataRecord/interface/L1RPCHsbConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"
#include "DataFormats/RPCDigi/interface/RPCDigiL1Link.h"
#include <memory>
#include <vector>

//class RPCTriggerGeo;

class RPCTrigger : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  explicit RPCTrigger(const edm::ParameterSet&);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------

  RPCConeBuilderFromES m_theLinksystemFromES;

  RPCPacManager<RPCPacData> m_pacManager;

  std::unique_ptr<RPCBasicTrigConfig> m_trigConfig;

  std::unique_ptr<RPCPacTrigger> m_pacTrigger;

  const int m_triggerDebug;
  unsigned long long m_cacheID;
  // TODO keep L1MuRegionalCandVec equally as RPCDigiL1LinkVec
  std::vector<L1MuRegionalCand> giveFinallCandindates(const L1RpcTBMuonsVec& finalMuons,
                                                      int type,
                                                      int bx,
                                                      edm::Handle<RPCDigiCollection> rpcDigis,
                                                      std::vector<RPCDigiL1Link>& retRPCDigiLink);

  const std::string m_label;
  const edm::EDGetTokenT<RPCDigiCollection> m_rpcDigiToken;

  const edm::EDPutTokenT<std::vector<L1MuRegionalCand>> m_brlCandPutToken;
  const edm::EDPutTokenT<std::vector<L1MuRegionalCand>> m_fwdCandPutToken;

  const edm::EDPutTokenT<std::vector<RPCDigiL1Link>> m_brlLinksPutToken;
  const edm::EDPutTokenT<std::vector<RPCDigiL1Link>> m_fwdLinksPutToken;
  const edm::ESGetToken<L1RPCConfig, L1RPCConfigRcd> m_configToken;
  const edm::ESGetToken<L1RPCConeBuilder, L1RPCConeBuilderRcd> m_coneBuilderToken;
  const edm::ESGetToken<L1RPCConeDefinition, L1RPCConeDefinitionRcd> m_coneDefinitionToken;
  const edm::ESGetToken<L1RPCHwConfig, L1RPCHwConfigRcd> m_hwConfigToken;
  const edm::ESGetToken<L1RPCBxOrConfig, L1RPCBxOrConfigRcd> m_BxOrConfigToken;
  const edm::ESGetToken<L1RPCHsbConfig, L1RPCHsbConfigRcd> m_hsbConfigToken;
};

#endif
