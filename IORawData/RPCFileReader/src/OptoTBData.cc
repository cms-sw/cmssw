#include "IORawData/RPCFileReader/interface/OptoTBData.h"

OptoTBData::OptoTBData(unsigned int defId, const rpcrawtodigi::EventRecords & event)
{ 
  theBX = event.recordBX().bx();
  std::pair<int,int> tctb = OptoTBData::getTCandTBNumbers( event.recordSLD().rmb(), defId);
  theTC = tctb.first;
  theTB = tctb.second;
  theOL = event.recordSLD().tbLinkInputNumber();
  theLMD.dat = event.recordCD().partitionData();
  theLMD.del = 0;
  theLMD.eod = event.recordCD().eod();
  theLMD.hp  = event.recordCD().halfP(); 
  theLMD.lb  = event.recordCD().lbInLink();
  theLMD.par = event.recordCD().partitionNumber();
}

OptoTBData::LMD::LMD(unsigned int rawData)
{
  unsigned int shift = 0;
  dat =  rawData &    0xff          ; shift += 8;
  par = (rawData &   0xf00) >> shift; shift += 4;
  del = (rawData &  0x7000) >> shift; shift += 3;
  eod = (rawData &  0x8000) >> shift; shift += 1;
  hp  = (rawData & 0x10000) >> shift; shift += 1;
  lb  = (rawData & 0x60000) >> shift; shift += 2;
}
unsigned int OptoTBData::LMD::raw() const
{
  unsigned int rawData = 0;
  unsigned int shift = 0;
  rawData = dat                                 ; shift += 8;
  rawData = rawData | ((par << shift) &   0xf00); shift += 4;
  rawData = rawData | ((del << shift )&  0x7000); shift += 3;
  rawData = rawData | ((eod << shift )&  0x8000); shift += 1;
  rawData = rawData | ((hp  << shift )& 0x10000); shift += 1;
  rawData = rawData | ((lb  << shift )& 0x60000); shift += 2;
  return rawData;
}

bool OptoTBData::LMD::operator<(const LMD & o) const
{
  return (raw()<o.raw());
//
//  if (del < o.del) return true;
//  if (del > o.del) return false;
//  if (lb < o.lb) return true;
//  if (lb > o.lb) return false;
//  if (par < o.par) return true;
//  if (par > o.par) return false;
//  if (hp < o.hp) return true;
//  if (hp > o.hp) return false;
//  if (eod < o.eod) return true;
//  if (eod > o.eod) return false;
//  if (raw < o.raw) return true;
//  return false;
//
}

bool OptoTBData::operator<(const OptoTBData & o) const
{
  if (bx() < o.bx()) return true;
  if (bx() > o.bx()) return false;
  if (tc() < o.tc()) return true;
  if (tc() > o.tc()) return false;
  if (tb() < o.tb()) return true;
  if (tb() > o.tb()) return false;
  if (ol() < o.ol()) return true;
  if (ol() > o.ol()) return false;
  return (theLMD < o.lmd());
}

std::pair<int,int> OptoTBData::getTCandTBNumbers(unsigned int rmb, unsigned int dcc) 
{
  int tcNumber = -1;
  int tbNumber = -1;
  for(unsigned int i=0;i<9;i++) if (rmb==i || rmb==9+i || rmb==18+i || rmb==27+i) tbNumber = i;
  for(unsigned int i=0;i<4;i++) if(rmb>=i*9 && rmb<9+i*9) tcNumber = i; //Count TC from 0, not from 1.
  tcNumber+=4*(792-dcc);
  return std::pair<int,int>(tcNumber,tbNumber);
}

