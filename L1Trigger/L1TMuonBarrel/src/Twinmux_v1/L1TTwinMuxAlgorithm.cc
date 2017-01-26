//-------------------------------------------------
//
//   Class: L1TTwinMuxAlgortithm
//
//   L1TTwinMuxAlgortithm
//
//
//   Author :
//   G. Flouris               U Ioannina    Oct. 2015
//--------------------------------------------------
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/L1ITMuonBarrelPrimitiveProducer.cc"
#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/L1TMuonTriggerPrimitiveProducer.cc"

#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/MBLTProducer.cc"
#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/MBLTCollection.h"
#include "L1Trigger/L1TMuonBarrel/src/Twinmux_v1/MBLTCollectionFwd.h"

#include <iostream>
#include <iomanip>

using namespace std;

class L1TTwinMuxAlgortithm  {
public:
  L1TTwinMuxAlgortithm();
  ~L1TTwinMuxAlgortithm() {}

  inline std::auto_ptr<L1MuDTChambPhContainer> produce( edm::Handle<L1MuDTChambPhContainer> phiDigis,
                                                        edm::Handle<L1MuDTChambThContainer> thetaDigis,
                                                        edm::Handle<RPCDigiCollection> rpcDigis,
                                                        const edm::EventSetup& c);

};

L1TTwinMuxAlgortithm::L1TTwinMuxAlgortithm() {

}


inline std::auto_ptr<L1MuDTChambPhContainer> L1TTwinMuxAlgortithm::produce(
                                                            edm::Handle<L1MuDTChambPhContainer> phiDigis,
                                                            edm::Handle<L1MuDTChambThContainer> thetaDigis,
                                                            edm::Handle<RPCDigiCollection> rpcDigis,
                                                            const edm::EventSetup& c) {


  TriggerPrimitiveCollection *l1tmtpp = new TriggerPrimitiveCollection();
  L1TMuonTPPproducer(phiDigis,thetaDigis,rpcDigis,l1tmtpp,c);

  std::shared_ptr<MBLTContainer> mblt = MBLTProducer(l1tmtpp);
  std::unique_ptr<L1ITMuonBarrelPrimitiveProducer> lmbpp ( new L1ITMuonBarrelPrimitiveProducer(mblt));
  std::auto_ptr<L1MuDTChambPhContainer> l1ttma = lmbpp->produce(c);

  delete l1tmtpp;

  return l1ttma;

}




