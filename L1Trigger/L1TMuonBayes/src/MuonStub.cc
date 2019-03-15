/*
 * MuonStub.cpp
 *
 *  Created on: Dec 21, 2018
 *      Author: kbunkow
 */


#include "L1Trigger/L1TMuonBayes/interface/MuonStub.h"
#include <iomanip>

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

MuonStub::MuonStub() {
  // TODO Auto-generated constructor stub

}

MuonStub::~MuonStub() {
  // TODO Auto-generated destructor stub
}

std::ostream & operator<< (std::ostream &out, const MuonStub &stub){
  out <<"MuonStub: ";
  out << " logicLayer: " <<std::setw(2)<< stub.logicLayer
      << " type: " <<std::setw(2)<< stub.type
      << " roll: " <<std::setw(1)<< stub.roll
      << " phiHw: " <<std::setw(5)<< stub.phiHw //<<" ("<<std::setw(8)<<stub.phi<<")"
      << " phiBHw: " <<std::setw(4)<< stub.phiBHw
      << " etaHw: " <<std::setw(4)<< stub.etaHw //<<" ("<<std::setw(8)<<stub.eta<<")"
      << " etaSigmaHw: " <<std::setw(3)<< stub.etaSigmaHw
      << " qualityHw: " <<std::setw(2)<< stub.qualityHw<<" "
      << " bx: " <<std::setw(1)<< stub.bx<<" "
      << " timing: " <<std::setw(2)<< stub.timing<<" "
      ;

  DetId detId(stub.detId);
  if (detId.det() != DetId::Muon){
    //std::cout << "PROBLEM: hit in unknown Det, detID: "<<detId.det()<<std::endl;
    return out;
  }

  switch (detId.subdetId()) {
  case MuonSubdetId::RPC: {
    RPCDetId rpcId(stub.detId);
    out <<" RPC "<< rpcId;
    break;
  }
  case MuonSubdetId::DT: {
    DTChamberId dtId(stub.detId);
    out <<" DT  "<< dtId;
    break;
  }
  case MuonSubdetId::CSC: {
    CSCDetId cscId(stub.detId);
    out <<" CSC "<< cscId;
    break;
  }
  }

  return out;
}
