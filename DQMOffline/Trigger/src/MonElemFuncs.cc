#include "DQMOffline/Trigger/interface/MonElemFuncs.h"

#include "DQMOffline/Trigger/interface/MonElemMgrEBEE.h"
#include "DQMOffline/Trigger/interface/CutCodes.h"
#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"
#include "DQMOffline/Trigger/interface/MonElemWithCutEBEE.h"

void MonElemFuncs::initStdEleHists(std::vector<MonElemManagerBase<EgHLTOffEle>*>& histVec,const std::string& baseName)
{
  histVec.push_back(new MonElemMgrEBEE<EgHLTOffEle,float>(baseName+"_et",baseName+" E_{T};E_{T} (GeV)",11,-10.,100.,&EgHLTOffEle::et));
  histVec.push_back(new MonElemMgrEBEE<EgHLTOffEle,float>(baseName+"_eta",baseName+" #eta;#eta",54,-2.7,2.7,&EgHLTOffEle::detEta));		
  histVec.push_back(new MonElemMgrEBEE<EgHLTOffEle,float>(baseName+"_phi",baseName+" #phi;#phi (rad)",50,-3.14,3.14,&EgHLTOffEle::phi));
  histVec.push_back(new MonElemMgrEBEE<EgHLTOffEle,int>(baseName+"_charge",baseName+" Charge; charge",2,-1.5,1.5,&EgHLTOffEle::charge));
  
  histVec.push_back(new MonElemMgrEBEE<EgHLTOffEle,float>(baseName+"_hOverE",baseName+" H/E; H/E",60,-0.05,0.25,&EgHLTOffEle::hOverE));
  histVec.push_back(new MonElemMgrEBEE<EgHLTOffEle,float>(baseName+"_dPhiIn",baseName+" #Delta #phi_{in}; #Delta #phi_{in}",50,-0.15,0.15,&EgHLTOffEle::dPhiIn));
  histVec.push_back(new MonElemMgrEBEE<EgHLTOffEle,float>(baseName+"_dEtaIn",baseName+" #Delta #eta_{in}; #Delta #eta_{in}",50,-0.02,0.02,&EgHLTOffEle::dEtaIn));
  histVec.push_back(new MonElemMgrEBEE<EgHLTOffEle,float>(baseName+"_sigmaEtaEta",baseName+" #sigma_{#eta#eta} ; #sigma_{#eta#eta}",60,-0.01,0.05,&EgHLTOffEle::sigmaEtaEta));
  histVec.push_back(new MonElemManager2D<EgHLTOffEle,float,float>(baseName+"_etaVsPhi",baseName+" #eta vs #phi;#eta;#phi (rad)",54,-2.7,2.7,50,-3.14,3.14,&EgHLTOffEle::detEta,&EgHLTOffEle::phi));
  //histVec.push_back(new MonElemManager2D<EgHLTOffEle,float,float>(baseName+"_sigEtaEtaVsEta",baseName+" #sigma_{#eta#eta} vs #eta;#eta;#sigma_{#eta#eta}",60,-0.01,0.05,54,-2.7,2.7,&EgHLTOffEle::sigmaEtaEta,&EgHLTOffEle::detEta));
}


void MonElemFuncs::initStdEffHists(std::vector<MonElemWithCutBase<EgHLTOffEle>*>& histVec,const std::string& baseName,int nrBins,double xMin,double xMax,float (EgHLTOffEle::*vsVarFunc)()const)
{
  //some convience typedefs, I hate typedefs but atleast here where they are defined is obvious
  typedef EgHLTDQMVarCut<EgHLTOffEle> VarCut;
  typedef MonElemWithCutEBEE<EgHLTOffEle,float> MonElemFloat;
  int stdCutCode = CutCodes::getCode("detEta:crack:sigmaEtaEta:hadem:dPhiIn:dEtaIn"); //will have it non hardcoded at a latter date
  //first do the zero and all cuts histograms
  histVec.push_back(new MonElemFloat(baseName+"_noCuts",baseName+" NoCuts",nrBins,xMin,xMax,vsVarFunc));
  histVec.push_back(new MonElemFloat(baseName+"_allCuts",baseName+" All Cuts",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(stdCutCode,&EgHLTOffEle::cutCode))); 
  //now the single histograms
  histVec.push_back(new MonElemFloat(baseName+"_single_hOverE",baseName+" Single H/E",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(CutCodes::HADEM,&EgHLTOffEle::cutCode)));
  histVec.push_back(new MonElemFloat(baseName+"_single_dEtaIn",baseName+" Single #Delta#eta_{in}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(CutCodes::DETAIN,&EgHLTOffEle::cutCode)));
  histVec.push_back(new MonElemFloat(baseName+"_single_dPhiIn",baseName+" Single #Delta#phi_{in}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(CutCodes::DPHIIN,&EgHLTOffEle::cutCode)));
  histVec.push_back(new MonElemFloat(baseName+"_single_sigmaEtaEta",baseName+" Single #sigma_{#eta#eta}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(CutCodes::SIGMAETAETA,&EgHLTOffEle::cutCode)));
  //now for the n-1
  histVec.push_back(new MonElemFloat(baseName+"_n1_dEtaIn",baseName+" N1 #Delta#eta_{in}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~CutCodes::DETAIN&stdCutCode,&EgHLTOffEle::cutCode)));
  histVec.push_back(new MonElemFloat(baseName+"_n1_dPhiIn",baseName+" N1 #Delta#phi_{in}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~CutCodes::DPHIIN&stdCutCode,&EgHLTOffEle::cutCode)));
  histVec.push_back(new MonElemFloat(baseName+"_n1_sigmaEtaEta",baseName+" N1 #sigma_{#eta#eta}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~CutCodes::SIGMAETAETA&stdCutCode,&EgHLTOffEle::cutCode)));
  histVec.push_back(new MonElemFloat(baseName+"_n1_hOverE",baseName+" N1 H/E",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~CutCodes::HADEM &stdCutCode,&EgHLTOffEle::cutCode)));
}

//we own the passed in cut, so we give it to the first mon element and then clone it after that
void MonElemFuncs::initStdEleCutHists(std::vector<MonElemWithCutBase<EgHLTOffEle>*>& histVec,const std::string& baseName,EgHLTDQMCut<EgHLTOffEle>* cut)
{
  histVec.push_back(new MonElemWithCutEBEE<EgHLTOffEle,float>(baseName+"_et",baseName+" E_{T};E_{T} (GeV)",11,-10.,100.,&EgHLTOffEle::et,cut));
  histVec.push_back(new MonElemWithCutEBEE<EgHLTOffEle,float>(baseName+"_eta",baseName+" #eta;#eta",12,-3,3,&EgHLTOffEle::detEta,cut ? cut->clone(): NULL));		
  histVec.push_back(new MonElemWithCutEBEE<EgHLTOffEle,float>(baseName+"_phi",baseName+" #phi;#phi (rad)",8,-3.14,3.14,&EgHLTOffEle::phi,cut ? cut->clone():NULL));
  histVec.push_back(new MonElemWithCutEBEE<EgHLTOffEle,int>(baseName+"_charge",baseName+" Charge; charge",2,-1.5,1.5,&EgHLTOffEle::charge,cut ? cut->clone():NULL));
  
 
}
