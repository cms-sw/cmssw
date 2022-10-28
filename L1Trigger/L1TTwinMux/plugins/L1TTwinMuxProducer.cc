//-------------------------------------------------
//
//   Class: L1TwinMuxProducer
//
//   L1TwinMuxProducer EDProducer
//
//
//   Author :
//   G. Flouris               U Ioannina    Feb. 2015
//   Mod.: g Karathanasis
//--------------------------------------------------

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondFormats/DataRecord/interface/L1TTwinMuxParamsRcd.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "L1Trigger/L1TTwinMux/interface/L1TTwinMuxAlgorithm.h"

#include <iostream>
#include <iomanip>
#include <memory>

using namespace std;

class L1TTwinMuxProducer : public edm::global::EDProducer<> {
public:
  L1TTwinMuxProducer(const edm::ParameterSet& pset);
  ~L1TTwinMuxProducer() override {}
  void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const override;

private:
  const edm::EDGetTokenT<L1MuDTChambPhContainer> m_dtdigi;
  const edm::EDGetTokenT<L1MuDTChambThContainer> m_dtthetadigi;
  const edm::EDGetTokenT<RPCDigiCollection> m_rpcsource;
  ///Event Setup Handler
  const edm::ESGetToken<L1TTwinMuxParams, L1TTwinMuxParamsRcd> m_tmParamsToken;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> m_rpcGeometryToken;

  const edm::EDPutTokenT<L1MuDTChambPhContainer> m_phContainerToken;
  const edm::EDPutTokenT<L1MuDTChambThContainer> m_thContainerToken;
};

L1TTwinMuxProducer::L1TTwinMuxProducer(const edm::ParameterSet& pset)
    : m_dtdigi(consumes(pset.getParameter<edm::InputTag>("DTDigi_Source"))),
      m_dtthetadigi(consumes(pset.getParameter<edm::InputTag>("DTThetaDigi_Source"))),
      m_rpcsource(consumes(pset.getParameter<edm::InputTag>("RPC_Source"))),
      m_tmParamsToken(esConsumes()),
      m_rpcGeometryToken(esConsumes()),
      m_phContainerToken(produces<L1MuDTChambPhContainer>()),
      m_thContainerToken(produces<L1MuDTChambThContainer>()) {}

void L1TTwinMuxProducer::produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const {
  ///Check consistency of the paramters
  auto const& tmParams = c.getData(m_tmParamsToken);

  ///Only RPC: the emulator's output consist from rpc->dy primitives only
  bool onlyRPC = tmParams.get_UseOnlyRPC();
  ///Only DT: the emulator's output consist from dt primitives only
  bool onlyDT = tmParams.get_UseOnlyDT();

  if (onlyDT && onlyRPC) {
    edm::LogWarning("Inconsistent configuration") << "onlyRPC and onlyDT options";
    return;
  }
  ///---Check consistency of the paramters

  edm::Handle<L1MuDTChambPhContainer> phiDigis = e.getHandle(m_dtdigi);
  edm::Handle<L1MuDTChambThContainer> thetaDigis = e.getHandle(m_dtthetadigi);

  edm::Handle<RPCDigiCollection> rpcDigis = e.getHandle(m_rpcsource);

  if (!phiDigis.isValid()) {
    edm::LogWarning("Inconsistent digis") << "input DT phi digis not valid";
  }

  auto const& rpcGeometry = c.getData(m_rpcGeometryToken);

  L1TTwinMuxAlgorithm l1tma;
  l1tma.run(phiDigis, thetaDigis, rpcDigis, tmParams, rpcGeometry);
  auto l1ttmp = l1tma.get_ph_tm_output();
  //null transfer of theta digis
  L1MuDTChambThContainer l1ttmth;
  const std::vector<L1MuDTChambThDigi>* theta = thetaDigis->getContainer();
  l1ttmth.setContainer(*theta);

  e.emplace(m_phContainerToken, std::move(l1ttmp));
  e.emplace(m_thContainerToken, std::move(l1ttmth));
}

DEFINE_FWK_MODULE(L1TTwinMuxProducer);
