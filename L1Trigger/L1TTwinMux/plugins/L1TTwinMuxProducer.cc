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
#include <FWCore/Framework/interface/ConsumesCollector.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "L1Trigger/L1TTwinMux/interface/L1TTwinMuxAlgorithm.h"

#include <iostream>
#include <iomanip>
#include <memory>

using namespace std;

//class L1TTwinMuxProducer: public edm::one::EDProducer<edm::one::SharedResources> 
//class L1TTwinMuxProducer: public edm::EDProducer
class  L1TTwinMuxProducer: public edm::stream::EDProducer<>
{
public:
  L1TTwinMuxProducer(const edm::ParameterSet & pset);
  ~L1TTwinMuxProducer() override {}
  void produce(edm::Event & e, const edm::EventSetup& c) override;
private:
  //L1TTwinMuxAlgorithm *  m_l1tma;
//  std::unique_ptr<L1TTwinMuxAlgorithm> m_l1tma(new L1TTwinMuxAlgorithm());
  edm::EDGetToken m_dtdigi, m_dtthetadigi, m_rpcsource;
  ///Event Setup Handler
  edm::ESHandle< L1TTwinMuxParams > tmParamsHandle;

};




L1TTwinMuxProducer::L1TTwinMuxProducer(const edm::ParameterSet & pset) {
//m_l1tma = new L1TTwinMuxAlgorithm();
// std::unique_ptr<L1TTwinMuxAlgorithm> m_l1tma(new L1TTwinMuxAlgorithm());

m_dtdigi      = consumes<L1MuDTChambPhContainer>(pset.getParameter<edm::InputTag>("DTDigi_Source"));
m_dtthetadigi = consumes<L1MuDTChambThContainer>(pset.getParameter<edm::InputTag>("DTThetaDigi_Source"));
m_rpcsource   = consumes<RPCDigiCollection>(pset.getParameter<edm::InputTag>("RPC_Source"));

produces<L1MuDTChambPhContainer>();
produces<L1MuDTChambThContainer>();

}

void L1TTwinMuxProducer::produce(edm::Event& e, const edm::EventSetup& c) {

  std::unique_ptr<L1TTwinMuxAlgorithm> m_l1tma(new L1TTwinMuxAlgorithm());
  ///Check consistency of the paramters
  const L1TTwinMuxParamsRcd& tmParamsRcd = c.get<L1TTwinMuxParamsRcd>();
  tmParamsRcd.get(tmParamsHandle);
  const L1TTwinMuxParams& tmParams = *tmParamsHandle.product();

  ///Only RPC: the emulator's output consist from rpc->dy primitives only
  bool onlyRPC = tmParams.get_UseOnlyRPC();
  ///Only DT: the emulator's output consist from dt primitives only
  bool onlyDT = tmParams.get_UseOnlyDT();

  if(onlyDT && onlyRPC) {edm::LogWarning("Inconsistent configuration")<<"onlyRPC and onlyDT options"; return;}
  ///---Check consistency of the paramters

  edm::Handle<L1MuDTChambPhContainer> phiDigis;
  edm::Handle<L1MuDTChambThContainer> thetaDigis;
  e.getByToken(m_dtdigi, phiDigis);
  e.getByToken(m_dtthetadigi, thetaDigis);

  edm::Handle<RPCDigiCollection> rpcDigis;
  e.getByToken(m_rpcsource, rpcDigis);

  if (! phiDigis.isValid()){
    edm::LogWarning("Inconsistent digis")<<"input DT phi digis not valid";
  }



  //std::unique_ptr<L1MuDTChambPhContainer> l1ttmp(new L1MuDTChambPhContainer);
  auto l1ttmp = std::make_unique<L1MuDTChambPhContainer>();
  m_l1tma->run(phiDigis, thetaDigis, rpcDigis,c);
  *l1ttmp = m_l1tma->get_ph_tm_output();
  //null transfer of theta digis
  auto l1ttmth = std::make_unique<L1MuDTChambThContainer>();
  const std::vector< L1MuDTChambThDigi>* theta=thetaDigis->getContainer();
  l1ttmth->setContainer(*theta); 

  e.put(std::move(l1ttmp));
  e.put(std::move(l1ttmth));
}



DEFINE_FWK_MODULE(L1TTwinMuxProducer);
