#ifndef DQMOFFLINE_TRIGGER_MONELEMFUNCS
#define DQMOFFLINE_TRIGGER_MONELEMFUNCS


//Author: Sam Harper

//Description: A collection of functions which assist and automate the creation
//             of useful monitor elements for Eg DQM
//

#include "DQMOffline/Trigger/interface/MonElemManager.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"
#include "DQMOffline/Trigger/interface/EgammaHLTEffSrc.h"
#include "DQMOffline/Trigger/interface/MonElemWithCut.h"
#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"

namespace MonElemFuncs {

  
  void initStdEleHists(std::vector<MonElemManagerBase<EgHLTOffEle>*>& histVec,const std::string& baseName); 
  void initStdEffHists(std::vector<MonElemWithCutBase<EgHLTOffEle>*>& histVec,const std::string& baseName,int nrBins,double xMin,double xMax,float (EgHLTOffEle::*vsVarFunc)()const);
  
  //we own the passed in pointer
  void initStdEleCutHists(std::vector<MonElemWithCutBase<EgHLTOffEle>*>& histVec,const std::string& baseName,EgHLTDQMCut<EgHLTOffEle>* cut=NULL);

}

#endif
