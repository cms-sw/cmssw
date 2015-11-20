//-------------------------------------------------
//
//   Class: L1TTwinMuxProducer
//
//   L1TTwinMuxProducer EDProducer
//
//
//   Author :
//   G. Flouris               U Ioannina    Oct. 2015
//--------------------------------------------------



#include "FWCore/Framework/interface/Event.h"
#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/L1TTwinMuxAlgorithm.cc"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <FWCore/Framework/interface/ConsumesCollector.h>
#include <FWCore/Framework/interface/one/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


#include <iostream>
#include <iomanip>

using namespace std;

class L1TTwinMuxProducer: public edm::one::EDProducer<edm::one::SharedResources> {
public:
  L1TTwinMuxProducer(const edm::ParameterSet & pset);
  ~L1TTwinMuxProducer() {}
  void produce(edm::Event & e, const edm::EventSetup& c);
private:
  L1TTwinMuxAlgortithm * m_l1tma;
  edm::EDGetToken m_dtdigi, m_dtthetadigi, m_rpcsource;

};




L1TTwinMuxProducer::L1TTwinMuxProducer(const edm::ParameterSet & pset) {
m_l1tma = new L1TTwinMuxAlgortithm();

m_dtdigi      = consumes<L1MuDTChambPhContainer>(pset.getParameter<edm::InputTag>("DTDigi_Source"));
m_dtthetadigi = consumes<L1MuDTChambThContainer>(pset.getParameter<edm::InputTag>("DTThetaDigi_Source"));
m_rpcsource   = consumes<RPCDigiCollection>(pset.getParameter<edm::InputTag>("RPC_Source"));

produces<L1MuDTChambPhContainer>();

}

void L1TTwinMuxProducer::produce(edm::Event& e, const edm::EventSetup& c) {

  edm::Handle<L1MuDTChambPhContainer> phiDigis;
  edm::Handle<L1MuDTChambThContainer> thetaDigis;
  e.getByToken(m_dtdigi, phiDigis);
  e.getByToken(m_dtthetadigi, thetaDigis);

  edm::Handle<RPCDigiCollection> rpcDigis;
  e.getByToken(m_rpcsource, rpcDigis);


  std::auto_ptr<L1MuDTChambPhContainer> l1ttmp = m_l1tma->produce(phiDigis, thetaDigis, rpcDigis,c);
  e.put(l1ttmp);
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TTwinMuxProducer);
