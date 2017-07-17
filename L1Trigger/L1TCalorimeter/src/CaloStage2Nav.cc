#include "L1Trigger/L1TCalorimeter/interface/CaloStage2Nav.h"

l1t::CaloStage2Nav::CaloStage2Nav():homePos_(0,0),currPos_(homePos_)
{

}

l1t::CaloStage2Nav::CaloStage2Nav(int iEta,int iPhi):homePos_(iEta,iPhi),currPos_(homePos_)
{
  
}

l1t::CaloStage2Nav::CaloStage2Nav(std::pair<int,int> pos):homePos_(pos),currPos_(homePos_)
{
  
}

std::pair<int,int> l1t::CaloStage2Nav::offsetFromCurrPos(int iEtaOffset,int iPhiOffset)const
{
  std::pair<int,int> offsetPos;
  offsetPos.first =  offsetIEta(currPos_.first,iEtaOffset);
  offsetPos.second = offsetIPhi(currPos_.second,iPhiOffset);
  return offsetPos;
}

std::pair<int,int> l1t::CaloStage2Nav::move(int iEtaOffset,int iPhiOffset)
{
  currPos_.first=offsetIEta(currPos_.first,iEtaOffset);
  currPos_.second=offsetIPhi(currPos_.second,iPhiOffset);
  return currPos_;
}
