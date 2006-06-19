//---------------------------------------------------------------------------
#include "L1Trigger/RPCTrigger/src/L1RpcTBMuon.h"
//#include "L1Trigger/RPCTrigger/src/L1RpcException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <sstream>
#include <iomanip>
#include <iostream>

using namespace std;
//---------------------------------------------------------------------------

unsigned int L1RpcTBMuon::ToBits(std::string where) const {
  if (where == "fsbIn") {
    return FSBIn::toBits(*this);
  }
  else if (where == "fsbOut") {
    return FSBOut::toBits(*this);
  }
  else {
  	//throw L1RpcException("unknown value of where: " + where);
    edm::LogError("RPCTrigger")<<"unknown value of where: " + where;
  } 
  return 0;
}

void L1RpcTBMuon::FromBits(std::string where, unsigned int value) {
  if (where == "fsbIn") {
    FSBIn::fromBits(*this, value);
  } 
  else if (where == "fsbOut") {
    FSBOut::fromBits(*this, value);
  }
  else {
  	//throw L1RpcException("unknown value of where: " + where);
    edm::LogError("RPCTrigger")<< "unknown value of where: " + where;
  }
}

unsigned int L1RpcTBMuon::FSBOut::toBits(const L1RpcTBMuon& muon) {
  unsigned int value = 0;
    	
  unsigned int shift = 0;
  unsigned int ptCode = (~(muon.PtCode)) & ptBitsMask;
  unsigned int quality = (~(muon.Quality)) & qualBitsMask;
  value = value &  muon.PhiAddress;         shift += phiBitsCnt;  
  value = value & (ptCode<<shift);          shift += ptBitsCnt;
  value = value & (quality<<shift);         shift += qualBitsCnt;
  value = value & (muon.EtaAddress<<shift); shift += etaBitsCnt + 1; //+1 beacouse H/F bits, unused in RPC
  value = value & (muon.Sign<<shift);       shift += signBitsCnt;
  
  return value;
}

void L1RpcTBMuon::FSBOut::fromBits(L1RpcTBMuon& muon, unsigned int value) {
  unsigned int shift = 0;
  muon.PhiAddress =  value & phiBitsMask;                    shift += phiBitsCnt;
  muon.PtCode     = (value & (ptBitsMask<<shift))   >> shift;  shift += ptBitsCnt;
  muon.Quality    = (value & (qualBitsMask<<shift)) >> shift;  shift += qualBitsCnt;
  muon.EtaAddress = (value & (etaBitsMask<<shift))  >> shift;  shift += etaBitsCnt + 1; //+1 beacouse H/F bits, unused in RPC
  muon.Sign       = (value & (signBitsMask<<shift)) >> shift;  shift += signBitsCnt; 
  
  muon.PtCode = (~(muon.PtCode)) & ptBitsMask;
  muon.Quality = (~(muon.Quality)) & qualBitsMask;
}

unsigned int L1RpcTBMuon::FSBIn::toBits(const L1RpcTBMuon& muon) {
  unsigned int value = 0;

  unsigned int shift = 0;
  value = value & (muon.Sign<<shift);       shift += signBitsCnt;
  value = value & (muon.PtCode<<shift);     shift += ptBitsCnt;
  value = value & (muon.Quality<<shift);    shift += qualBitsCnt;
  value = value & (muon.PhiAddress<<shift); shift += phiBitsCnt;
  value = value & (muon.EtaAddress<<shift); shift += etaBitsCnt; 
   
  return value;
}

void L1RpcTBMuon::FSBIn::fromBits(L1RpcTBMuon& muon, unsigned int value) {
  unsigned int shift = 0;
  muon.Sign       = (value & (signBitsMask<<shift)) >> shift;  shift += signBitsCnt;
  muon.PtCode     = (value & (ptBitsMask<<shift))   >> shift;  shift += ptBitsCnt;
  muon.Quality    = (value & (qualBitsMask<<shift)) >> shift;  shift += qualBitsCnt;
  muon.PhiAddress = (value & (phiBitsMask<<shift))  >> shift;  shift += phiBitsCnt;
  muon.EtaAddress = (value & (etaBitsMask<<shift))  >> shift;  shift += etaBitsCnt;
}

std::string L1RpcTBMuon::BitsToString() const {
  ostringstream ostr;
  ostr<<"qu "<<Quality<<", pt "<<setw(2)<<PtCode<<", sig "<<Sign
      <<", phi "<<setw(3)<<PhiAddress<<", eta "<<setw(2)<<EtaAddress;
  return ostr.str();
}
