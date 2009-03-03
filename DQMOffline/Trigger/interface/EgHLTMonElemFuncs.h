#ifndef DQMOFFLINE_TRIGGER_EGHLTMONELEMFUNCS
#define DQMOFFLINE_TRIGGER_EGHLTMONELEMFUNCS


//Author: Sam Harper

//Description: A collection of functions which assist and automate the creation
//             of useful monitor elements for Eg DQM
//

#include "DQMOffline/Trigger/interface/EgHLTMonElemManager.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"
#include "DQMOffline/Trigger/interface/EgHLTOffPho.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemWithCut.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemMgrEBEE.h"
#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemContainer.h"
#include "DQMOffline/Trigger/interface/EgHLTBinData.h"

namespace egHLT {

  class CutMasks;

  namespace MonElemFuncs {
    
    
    void initStdEleHists(std::vector<MonElemManagerBase<OffEle>*>& histVec,const std::string& baseName,const BinData& bins); 
    void initStdPhoHists(std::vector<MonElemManagerBase<OffPho>*>& histVec,const std::string& baseName,const BinData& bins); 
    void initStdEffHists(std::vector<MonElemWithCutBase<OffEle>*>& histVec,const std::string& baseName,int nrBins,double xMin,double xMax,float (OffEle::*vsVarFunc)()const,const CutMasks& masks); 
    void initStdEffHists(std::vector<MonElemWithCutBase<OffEle>*>& histVec,const std::string& baseName,const BinData::Data1D& bins,float (OffEle::*vsVarFunc)()const,const CutMasks& masks);
    void initStdEffHists(std::vector<MonElemWithCutBase<OffPho>*>& histVec,const std::string& baseName,int nrBins,double xMin,double xMax,float (OffPho::*vsVarFunc)()const,const CutMasks& masks);   
    void initStdEffHists(std::vector<MonElemWithCutBase<OffPho>*>& histVec,const std::string& baseName,const BinData::Data1D& bins,float (OffPho::*vsVarFunc)()const,const CutMasks& masks);

    //we own the passed in pointer
    void initStdEleCutHists(std::vector<MonElemWithCutBase<OffEle>*>& histVec,const std::string& baseName,const BinData& bins,EgHLTDQMCut<OffEle>* cut=NULL);


    
  
    void initTightLooseTrigHists(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins,EgHLTDQMCut<OffEle>* eleCut);
    void initTightLooseTrigHists(std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins,EgHLTDQMCut<OffPho>* phoCut);
    
    void initTightLooseTrigHistsTrigCuts(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins);  
    void initTightLooseTrigHistsTrigCuts(std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins);
    
    void addTightLooseTrigHist(std::vector<MonElemContainer<OffEle>*>& eleMonElems,
			       const std::string& tightTrig,const std::string& looseTrig,
			       EgHLTDQMCut<OffEle>* eleCut,const std::string& histId,const BinData& bins);


    void addTightLooseTrigHist(std::vector<MonElemContainer<OffPho>*>& phoMonElems,
			       const std::string& tightTrig,const std::string& looseTrig,
			       EgHLTDQMCut<OffPho>* phoCut,const std::string& histId,const BinData& bins);
    

    
    void initTightLooseDiObjTrigHistsTrigCuts(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins);
    void initTightLooseDiObjTrigHistsTrigCuts(std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins);

    //ele only
    void initTrigTagProbeHists(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string> filterNames,const BinData& bins);
  

    template<class T,typename varType> void addStdHist(std::vector<MonElemManagerBase<T>*>& histVec,const std::string& name,const std::string& title,
						       const BinData::Data1D& binData,varType (T::*varFunc)()const){
      histVec.push_back(new MonElemMgrEBEE<T,varType>(name,title,binData.nr,binData.min,binData.max,varFunc));
    }
  }
}
#endif
