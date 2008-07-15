#ifndef  DQMOFFLINE_TRIGGER_EGHLTOFFELESEL
#define  DQMOFFLINE_TRIGGER_EGHLTOFFELESEL

//this class works out which cuts the electron passes/fails
#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"
#include "DQMOffline/Trigger/interface/CutValues.h"

#include <iostream>

class EgHLTOffEleSel  {

 private:

  std::vector<CutValues> cutValues_;
  
 public:
  EgHLTOffEleSel();  
  EgHLTOffEleSel(const EgHLTOffEleSel& rhs):cutValues_(rhs.cutValues_){}
  ~EgHLTOffEleSel(){} 

  EgHLTOffEleSel& operator=(const EgHLTOffEleSel& rhs){cutValues_=rhs.cutValues_;return *this;}

  bool passCuts(const EgHLTOffEle& ele,int cutMask=~0x0)const{return getCutCode(ele,cutMask)==0x0;}
  int getCutCode(const EgHLTOffEle& ele,int cutMask=~0x0)const;
  
  static int getCutCode(const EgHLTOffEle& ele,const CutValues& cuts,int cutMask=~0x0);

  void addCuts(const CutValues& cuts){cutValues_.push_back(cuts);}
  CutValues& getCuts(int type); //gets the cuts appropriate to the type of the electron
  const CutValues& getCuts(int type)const;
  CutValues& getCutsByIndx(int cutNr){return cutValues_[cutNr];} 
  const CutValues& getCutsByIndx(int cutNr)const{return cutValues_[cutNr];}
  int nrCuts()const{return cutValues_.size();}
  
  void setHighNrgy();
  void setPreSel();
  void setPreSelWithEp();
  void setCutMask(int cutMask,int eleType=~0x0);
  void removeCuts(int cutCode,int eleType=~0x0);
  void setMinEt(float minEt,int eleType=~0x0);
 
  void clearCuts(){cutValues_.clear();}
 
};

#endif
