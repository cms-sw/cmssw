//---------------------------------------------------------------------------
#include "L1Trigger/RPCTrigger/interface/RPCTBMuon.h"
#ifndef _STAND_ALONE
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif  // _STAND_ALONE
#include "L1Trigger/RPCTrigger/interface/RPCException.h"

#include <sstream>
#include <iomanip>
#include <iostream>

using namespace std;
//---------------------------------------------------------------------------
RPCTBMuon::RPCTBMuon() : RPCMuon() {
  m_Killed = false;

  m_GBData = 0;

  m_EtaAddress = 0;
  m_PhiAddress = 0;

  m_muonBitsType = mbtUnset;
}
//---------------------------------------------------------------------------
RPCTBMuon::RPCTBMuon(int ptCode, int quality, int sign, int patternNum, unsigned short firedPlanes)
    : RPCMuon(ptCode, quality, sign, patternNum, firedPlanes) {
  m_Killed = false;

  m_GBData = 0;

  m_EtaAddress = 0;
  m_PhiAddress = 0;

  m_muonBitsType = mbtUnset;
}

RPCTBMuon::RPCTBMuon(int ptCode, int quality, int sign, MuonBitsType muonBitsType)
    : RPCMuon(ptCode, quality, sign, 0, 0) {
  m_Killed = false;

  m_GBData = 0;

  m_EtaAddress = 0;
  m_PhiAddress = 0;

  m_muonBitsType = muonBitsType;
}

/**
*
* \brief Gives debuging info in human readable format (1) or technicall format (2)
* \note Possible debugFormat codes (1,2) are defined in RPCTrigger constructor
*
*/
std::string RPCTBMuon::printDebugInfo(int debugFormat) const {
  std::ostringstream sDebugInfo;
  if (debugFormat == 1) {  // Human readable

    sDebugInfo << "TBMuon: code: " << getPtCode() << " etaAddr: " << getEtaAddr() << " phiAddr: " << getPhiAddr()
               << " sgAddr: " << getSegmentAddr() << " scAddr: " << getSectorAddr() << " gbData: " << getGBData();

  } else {  //technicall
    sDebugInfo << "TBMuon pt " << getPtCode() << " ql " << getQuality() << " sgn " << getSign() << " tw " << getTower()
               << " sc " << getLogSector() << " sg " << getLogSegment() << " phi " << getPhiAddr() << " eta "
               << getEtaAddr() << " gbD " << getGBData() << " bits " << toBits(mbtFSBOut);
  }

  return sDebugInfo.str();
}
//---------------------------------------------------------------------------
// Simple setters and getters

///Combined quality and ptCode, 8 bits [7...6 m_Quality, 5...1 m_PtCode, 0 sign], used in GhoustBusters
int RPCTBMuon::getCode() const {
  // 1 bit, 0 - negative, 1 - positive.
  unsigned int signLocal = 0;
  if (m_Sign == 0)
    signLocal = 1;  // invert
  return (m_Quality << 6 | m_PtCode << 1 | signLocal);
}

///Sets combined code: 8 bits [7...5 m_Quality, 4...0 m_PtCode].
void RPCTBMuon::setCode(int code) {
  m_Quality = (code & (3 << 5)) >> 5;
  m_PtCode = code & 31;
}

void RPCTBMuon::setPhiAddr(int phiAddr) { m_PhiAddress = phiAddr; }

void RPCTBMuon::setSectorAddr(int sectorAddr) { m_PhiAddress = m_PhiAddress | sectorAddr << 4; }

void RPCTBMuon::setEtaAddr(int etaAddr) { m_EtaAddress = etaAddr; }

void RPCTBMuon::setAddress(int etaAddr, int phiAddr) {
  m_EtaAddress = etaAddr;
  m_PhiAddress = phiAddr;
}

void RPCTBMuon::setAddress(int tbNumber, int tbTower, int phiAddr) {
  m_EtaAddress = (tbNumber << 2) | tbTower;
  m_PhiAddress = phiAddr;
}

void RPCTBMuon::setGBData(unsigned int gbData) { m_GBData = gbData; }

int RPCTBMuon::getEtaAddr() const { return m_EtaAddress; }

int RPCTBMuon::getPhiAddr() const { return m_PhiAddress; }

int RPCTBMuon::getSegmentAddr() const { return m_PhiAddress & 15; }

int RPCTBMuon::getSectorAddr() const { return (m_PhiAddress & 0xF0) >> 4; }

int RPCTBMuon::getContinSegmAddr() const { return getSectorAddr() * 12 + getSegmentAddr(); }

void RPCTBMuon::setCodeAndPhiAddr(int code, int phiAddr) {
  setCode(code);
  m_PhiAddress = phiAddr;
}

void RPCTBMuon::setCodeAndEtaAddr(int code, int etaAddr) {
  setCode(code);
  m_EtaAddress = etaAddr;
}

int RPCTBMuon::getGBData() const { return m_GBData; }

std::string RPCTBMuon::getGBDataBitStr() const {
  std::string str = "00";
  if (m_GBData == 1)
    str = "01";
  else if (m_GBData == 2)
    str = "10";
  else if (m_GBData == 3)
    str = "11";
  return str;
}

void RPCTBMuon::setGBDataKilledFirst() { m_GBData = m_GBData | 1; }

void RPCTBMuon::setGBDataKilledLast() { m_GBData = m_GBData | 2; }

bool RPCTBMuon::gBDataKilledFirst() const { return (m_GBData & 1); }

bool RPCTBMuon::gBDataKilledLast() const { return (m_GBData & 2); }

//---------------------------------------------------------------------------
void RPCTBMuon::kill() { m_Killed = true; }

/** @return true = was non-empty muon and was killed
  * false = was not killed or is zero */
bool RPCTBMuon::wasKilled() const {
  if (m_PtCode > 0 && m_Killed)
    return true;
  else
    return false;
}

/** @return true = was no-zero muon and was not killed
  * false = is killed or is zero */
bool RPCTBMuon::isLive() const {
  if (m_PtCode > 0 && !m_Killed)
    return true;
  else
    return false;
}

//---------------------------------------------------------------------------
#ifndef _STAND_ALONE
RPCTBMuon::RPCTBMuon(const RPCPacMuon& pacMuon) : RPCMuon(pacMuon) {
  m_Killed = false;

  m_GBData = 0;

  m_EtaAddress = 0;
  m_PhiAddress = 0;
  m_muonBitsType = mbtUnset;
}
#endif  //_STAND_ALONE
//---------------------------------------------------------------------------

unsigned int RPCTBMuon::toBits(RPCTBMuon::MuonBitsType muonBitsType) const {
  if (muonBitsType == mbtPACOut) {
    return PACOut::toBits(*this);
  } else if (muonBitsType == mbtTBSortOut) {
    return TBOut::toBits(*this);
  } else if (muonBitsType == mbtTCSortOut) {
    return TCOut::toBits(*this);
  } else if (muonBitsType == mbtHSBOut) {
    return HSBOut::toBits(*this);
  } else if (muonBitsType == mbtFSBOut) {
    return FSBOut::toBits(*this);
  } else {
    throw RPCException("unknown value of muonBitsType");
    //edm::LogError("RPCTrigger")<<"unknown value of where: " + where;
  }
  return 0;
}

void RPCTBMuon::fromBits(RPCTBMuon::MuonBitsType muonBitsType, unsigned int value) {
  if (muonBitsType == mbtPACOut) {
    PACOut::fromBits(*this, value);
  } else if (muonBitsType == mbtTBSortOut) {
    TBOut::fromBits(*this, value);
  } else if (muonBitsType == mbtTCSortOut) {
    TCOut::fromBits(*this, value);
  } else if (muonBitsType == mbtHSBOut) {
    HSBOut::fromBits(*this, value);
  } else if (muonBitsType == mbtFSBOut) {
    FSBOut::fromBits(*this, value);
  } else {
    RPCException("unknown value of muonBitsType");
    //edm::LogError("RPCTrigger")<< "unknown value of where: " + where;
  }
}

void RPCTBMuon::PACOut::fromBits(RPCTBMuon& muon, unsigned int value) {
  unsigned int shift = 0;
  muon.m_Sign = (value & (m_signBitsMask << shift)) >> shift;
  shift += m_signBitsCnt;
  muon.m_PtCode = (value & (m_ptBitsMask << shift)) >> shift;
  shift += m_ptBitsCnt;
  muon.m_Quality = (value & (m_qualBitsMask << shift)) >> shift;
  shift += m_qualBitsCnt;
}

unsigned int RPCTBMuon::PACOut::toBits(const RPCTBMuon& muon) {
  unsigned int value = 0;
  unsigned int shift = 0;
  value = (muon.m_Sign << shift);
  shift += m_signBitsCnt;
  value = value | (muon.m_PtCode << shift);
  shift += m_ptBitsCnt;
  value = value | (muon.m_Quality << shift);
  shift += m_qualBitsCnt;

  return value;
}

//-----------------------
void RPCTBMuon::TBOut::fromBits(RPCTBMuon& muon, unsigned int value) {
  unsigned int shift = 0;
  muon.m_Sign = (value & (m_signBitsMask << shift)) >> shift;
  shift += m_signBitsCnt;
  muon.m_PtCode = (value & (m_ptBitsMask << shift)) >> shift;
  shift += m_ptBitsCnt;
  muon.m_Quality = (value & (m_qualBitsMask << shift)) >> shift;
  shift += m_qualBitsCnt;
  muon.m_PhiAddress = (value & (m_phiBitsMask << shift)) >> shift;
  shift += m_phiBitsCnt;
  muon.m_EtaAddress = (value & (m_etaBitsMask << shift)) >> shift;
  shift += m_etaBitsCnt;
  muon.m_GBData = (value & (m_gbDataBitsMask << shift)) >> shift;
  shift += m_gbDataBitsCnt;
}

unsigned int RPCTBMuon::TBOut::toBits(const RPCTBMuon& muon) {
  unsigned int value = 0;
  unsigned int shift = 0;
  value = (muon.m_Sign << shift);
  shift += m_signBitsCnt;
  value = value | (muon.m_PtCode << shift);
  shift += m_ptBitsCnt;
  value = value | (muon.m_Quality << shift);
  shift += m_qualBitsCnt;
  value = value | (muon.m_PhiAddress << shift);
  shift += m_phiBitsCnt;
  value = value | (muon.m_EtaAddress << shift);
  shift += m_etaBitsCnt;
  value = value | (muon.m_GBData << shift);
  shift += m_gbDataBitsCnt;
  return value;
}

//-----------------------
void RPCTBMuon::TCOut::fromBits(RPCTBMuon& muon, unsigned int value) {
  unsigned int shift = 0;
  muon.m_Sign = (value & (m_signBitsMask << shift)) >> shift;
  shift += m_signBitsCnt;
  muon.m_PtCode = (value & (m_ptBitsMask << shift)) >> shift;
  shift += m_ptBitsCnt;
  muon.m_Quality = (value & (m_qualBitsMask << shift)) >> shift;
  shift += m_qualBitsCnt;
  muon.m_PhiAddress = (value & (m_phiBitsMask << shift)) >> shift;
  shift += m_phiBitsCnt;
  muon.m_EtaAddress = (value & (m_etaBitsMask << shift)) >> shift;
  shift += m_etaBitsCnt;
  muon.m_GBData = (value & (m_gbDataBitsMask << shift)) >> shift;
  shift += m_gbDataBitsCnt;
}

unsigned int RPCTBMuon::TCOut::toBits(const RPCTBMuon& muon) {
  unsigned int value = 0;
  unsigned int shift = 0;
  value = (muon.m_Sign << shift);
  shift += m_signBitsCnt;
  value = value | (muon.m_PtCode << shift);
  shift += m_ptBitsCnt;
  value = value | (muon.m_Quality << shift);
  shift += m_qualBitsCnt;
  value = value | (muon.m_PhiAddress << shift);
  shift += m_phiBitsCnt;
  value = value | (muon.m_EtaAddress << shift);
  shift += m_etaBitsCnt;
  value = value | (muon.m_GBData << shift);
  shift += m_gbDataBitsCnt;
  return value;
}
//-------------------------------

unsigned int RPCTBMuon::HSBOut::toBits(const RPCTBMuon& muon) {
  unsigned int value = 0;

  unsigned int shift = 0;
  value = value | (muon.m_Sign << shift);
  shift += m_signBitsCnt;
  // value = (muon.m_Sign<<shift);       shift += m_signBitsCnt;
  value = value | (muon.m_PtCode << shift);
  shift += m_ptBitsCnt;
  value = value | (muon.m_Quality << shift);
  shift += m_qualBitsCnt;
  value = value | (muon.m_PhiAddress << shift);
  shift += m_phiBitsCnt;
  value = value | (muon.m_EtaAddress << shift);
  shift += m_etaBitsCnt;

  return value;
}

void RPCTBMuon::HSBOut::fromBits(RPCTBMuon& muon, unsigned int value) {
  unsigned int shift = 0;
  muon.m_Sign = (value & (m_signBitsMask << shift)) >> shift;
  shift += m_signBitsCnt;
  muon.m_PtCode = (value & (m_ptBitsMask << shift)) >> shift;
  shift += m_ptBitsCnt;
  muon.m_Quality = (value & (m_qualBitsMask << shift)) >> shift;
  shift += m_qualBitsCnt;
  muon.m_PhiAddress = (value & (m_phiBitsMask << shift)) >> shift;
  shift += m_phiBitsCnt;
  muon.m_EtaAddress = (value & (m_etaBitsMask << shift)) >> shift;
  shift += m_etaBitsCnt;
}

unsigned int RPCTBMuon::FSBOut::toBits(const RPCTBMuon& muon) {
  unsigned int value = 0;

  unsigned int shift = 0;
  //bita are reversed, to be zero when cable not connected
  unsigned int ptCode = (~(muon.m_PtCode)) & m_ptBitsMask;
  unsigned int quality = (~(muon.m_Quality)) & m_qualBitsMask;

  value = value | muon.m_PhiAddress;
  shift += m_phiBitsCnt;
  value = value | (ptCode << shift);
  shift += m_ptBitsCnt;
  value = value | (quality << shift);
  shift += m_qualBitsCnt;

  //+1 beacouse H/F bits, unused in RPC:
  value = value | (muon.m_EtaAddress << shift);
  shift += m_etaBitsCnt + 1;
  value = value | (muon.m_Sign << shift);
  shift += m_signBitsCnt;

  return value;
}

void RPCTBMuon::FSBOut::fromBits(RPCTBMuon& muon, unsigned int value) {
  unsigned int shift = 0;
  muon.m_PhiAddress = value & m_phiBitsMask;
  shift += m_phiBitsCnt;
  muon.m_PtCode = (value & (m_ptBitsMask << shift)) >> shift;
  shift += m_ptBitsCnt;
  muon.m_Quality = (value & (m_qualBitsMask << shift)) >> shift;
  shift += m_qualBitsCnt;

  //+1 beacouse H/F bits, unused in RPC:
  muon.m_EtaAddress = (value & (m_etaBitsMask << shift)) >> shift;
  shift += m_etaBitsCnt + 1;
  muon.m_Sign = (value & (m_signBitsMask << shift)) >> shift;
  shift += m_signBitsCnt;

  muon.m_PtCode = (~(muon.m_PtCode)) & m_ptBitsMask;
  muon.m_Quality = (~(muon.m_Quality)) & m_qualBitsMask;
}

std::string RPCTBMuon::toString(int format) const {
  ostringstream ostr;
  if (format == 0) {
    ostr << "qu " << m_Quality << ", pt " << setw(2) << m_PtCode << ", sig " << m_Sign << ", phi " << setw(3)
         << m_PhiAddress << ", eta " << setw(2) << m_EtaAddress;
  } else if (format == 1) {
    ostr << " " << m_Quality << " " << setw(2) << m_PtCode << " " << m_Sign << " " << setw(3) << m_PhiAddress << " "
         << setw(2) << m_EtaAddress;
  } else if (format == 2) {
    ostr << " " << m_Quality << " " << setw(2) << m_PtCode << " " << m_Sign;
  }

  return ostr.str();
}
