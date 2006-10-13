//---------------------------------------------------------------------------
#include "L1Trigger/RPCTrigger/src/L1RpcTBMuon.h"
//#include "L1Trigger/RPCTrigger/src/L1RpcException.h"
#ifndef _STAND_ALONE
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif // _STAND_ALONE
#include "L1Trigger/RPCTrigger/src/RPCException.h"

#include <sstream>
#include <iomanip>
#include <iostream>

using namespace std;
//---------------------------------------------------------------------------
L1RpcTBMuon::L1RpcTBMuon(): L1RpcMuon() {
    Killed = false;

    GBData = 0;

    EtaAddress = 0;
    PhiAddress = 0;
}
//---------------------------------------------------------------------------
L1RpcTBMuon::L1RpcTBMuon(int ptCode, int quality, int sign,
                         int patternNum, unsigned short firedPlanes):
    L1RpcMuon(ptCode, quality, sign, patternNum, firedPlanes) 
{
    Killed = false;

    GBData = 0;

    EtaAddress = 0;
    PhiAddress = 0;
}

/**
*
* \brief Gives debuging info in human readable format (1) or technicall format (2)
* \note Possible debugFormat codes (1,2) are defined in RPCTrigger constructor
*
*/
std::string L1RpcTBMuon::printDebugInfo(int debugFormat) const{

  std::ostringstream sDebugInfo;
  if (debugFormat==1){  // Human readable

    sDebugInfo << "TBMuon: code: " << GetPtCode()
               << " etaAddr: " << GetEtaAddr()
               << " phiAddr: " << GetPhiAddr()
               << " sgAddr: " << GetSegmentAddr()
               << " scAddr: " << GetSectorAddr()
               << " gbData: " << GetGBData();

  }
  else {        //technicall
   sDebugInfo << "TBMuon pt "<< GetPtCode() 
              <<   " ql " <<GetQuality() 
              <<   " sgn " << GetSign()
              <<   " tw " << GetTower()
              <<   " sc " << GetLogSector()
              <<   " sg " << GetLogSegment()
              <<   " bits " << ToBits("fsbOut");
  }

  return sDebugInfo.str();

}
//---------------------------------------------------------------------------
// Simple setters and getters

///Combined quality and ptCode, 8 bits [7...5 Quality, 4...0 PtCode], used in GhoustBusters
int L1RpcTBMuon::GetCode() const {  return (Quality<<5 | PtCode); }

///Sets combined code: 8 bits [7...5 Quality, 4...0 PtCode].
void L1RpcTBMuon::SetCode(int code) {
    Quality = (code & (3<<5))>>5;
    PtCode = code & 31;
}


void L1RpcTBMuon::SetPhiAddr(int phiAddr) { PhiAddress = phiAddr;}

void L1RpcTBMuon::SetSectorAddr(int sectorAddr){ PhiAddress = PhiAddress | sectorAddr<<4;}

void L1RpcTBMuon::SetEtaAddr(int etaAddr) { EtaAddress = etaAddr;}
  
void L1RpcTBMuon::SetAddress(int etaAddr, int phiAddr) { 
     EtaAddress = etaAddr;
     PhiAddress = phiAddr;
}

void L1RpcTBMuon::SetAddress(int tbNumber, int tbTower, int phiAddr) {
    EtaAddress = (tbNumber<<2) | tbTower;
    PhiAddress = phiAddr;
}

int L1RpcTBMuon::GetEtaAddr() const { return EtaAddress; }

int L1RpcTBMuon::GetPhiAddr() const { return PhiAddress; }

int L1RpcTBMuon::GetSegmentAddr() const { return PhiAddress & 15; }

int L1RpcTBMuon::GetSectorAddr() const { return (PhiAddress & 0xF0)>>4; }

int L1RpcTBMuon::GetContinSegmAddr() const { return GetSectorAddr()*12 + GetSegmentAddr();}

void L1RpcTBMuon::SetCodeAndPhiAddr(int code, int phiAddr) {
    SetCode(code);
    PhiAddress = phiAddr;
}

void L1RpcTBMuon::SetCodeAndEtaAddr(int code, int etaAddr) {
    SetCode(code);
    EtaAddress = etaAddr;
}
  
int L1RpcTBMuon::GetGBData() const { return GBData;}

std::string L1RpcTBMuon::GetGBDataBitStr() const {
    std::string str = "00";
    if (GBData == 1)
      str = "01";
    else if (GBData == 2)
      str = "10";
    else if (GBData == 3)
      str = "11";
    return str;  
}

void L1RpcTBMuon::SetGBDataKilledFirst() { GBData = GBData | 1;}

void L1RpcTBMuon::SetGBDataKilledLast() { GBData = GBData | 2; }

bool L1RpcTBMuon::GBDataKilledFirst() const { return (GBData & 1);}

bool L1RpcTBMuon::GBDataKilledLast() const { return (GBData & 2);}


//---------------------------------------------------------------------------
void L1RpcTBMuon::Kill() { Killed = true; }

/** @return true = was non-empty muon and was killed
  * false = was not killed or is zero */
bool L1RpcTBMuon::WasKilled() const {
    if(PtCode > 0 && Killed)
      return true;
    else return false;
}

/** @return true = was no-zero muon and was not killed
  * false = is killed or is zero */
bool L1RpcTBMuon::IsLive() const {
    if(PtCode > 0 && !Killed)
      return true;
    else return false;
}

//---------------------------------------------------------------------------
L1RpcTBMuon::L1RpcTBMuon(const L1RpcPacMuon& pacMuon):
    L1RpcMuon(pacMuon) 
{
    Killed = false;

    GBData = 0;

    EtaAddress = 0;
    PhiAddress = 0;
}
//---------------------------------------------------------------------------
unsigned int L1RpcTBMuon::ToBits(std::string where) const {
  if (where == "fsbIn") {
    return FSBIn::toBits(*this);
  }
  else if (where == "fsbOut") {
    return FSBOut::toBits(*this);
  }
  else {
    throw L1RpcException("unknown value of where: " + where);
    //edm::LogError("RPCTrigger")<<"unknown value of where: " + where;
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
    throw L1RpcException("unknown value of where: " + where);
    //edm::LogError("RPCTrigger")<< "unknown value of where: " + where;
  }
}

unsigned int L1RpcTBMuon::FSBOut::toBits(const L1RpcTBMuon& muon) {
  unsigned int value = 0;
    	
  unsigned int shift = 0;
  unsigned int ptCode = (~(muon.PtCode)) & ptBitsMask;
  unsigned int quality = (~(muon.Quality)) & qualBitsMask;
  value = value |  muon.PhiAddress;         shift += phiBitsCnt;  
  //  value = muon.PhiAddress;         shift += phiBitsCnt;  
  value = value | (ptCode<<shift);          shift += ptBitsCnt;
  value = value | (quality<<shift);         shift += qualBitsCnt;
  value = value | (muon.EtaAddress<<shift); shift += etaBitsCnt + 1; //+1 beacouse H/F bits, unused in RPC
  value = value | (muon.Sign<<shift);       shift += signBitsCnt;
  
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
  value = value | (muon.Sign<<shift);       shift += signBitsCnt;
 // value = (muon.Sign<<shift);       shift += signBitsCnt;
  value = value | (muon.PtCode<<shift);     shift += ptBitsCnt;
  value = value | (muon.Quality<<shift);    shift += qualBitsCnt;
  value = value | (muon.PhiAddress<<shift); shift += phiBitsCnt;
  value = value | (muon.EtaAddress<<shift); shift += etaBitsCnt; 
   
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
