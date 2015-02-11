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
#include "DQMOffline/Trigger/interface/EgHLTTrigTools.h"
#include "DQMOffline/Trigger/interface/EgHLTMonElemWithCutEBEE.h"

#include <boost/algorithm/string.hpp>

namespace egHLT {

  struct CutMasks;

  class MonElemFuncs {
   private:
    DQMStore::IBooker& iBooker;
    const TrigCodes& trigCodes;

   public:
    MonElemFuncs(DQMStore::IBooker& i, const TrigCodes& c): iBooker(i), trigCodes(c) {};
    ~MonElemFuncs() {};
    DQMStore::IBooker& getIB() { return iBooker; };

   public:  
    
    void initStdEleHists(std::vector<MonElemManagerBase<OffEle>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData& bins); 
    void initStdPhoHists(std::vector<MonElemManagerBase<OffPho>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData& bins); 
    void initStdEffHists(std::vector<MonElemWithCutBase<OffEle>*>& histVec,const std::string& filterName,const std::string& baseName,int nrBins,double xMin,double xMax,float (OffEle::*vsVarFunc)()const,const CutMasks& masks); 
    void initStdEffHists(std::vector<MonElemWithCutBase<OffEle>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData::Data1D& bins,float (OffEle::*vsVarFunc)()const,const CutMasks& masks);
    void initStdEffHists(std::vector<MonElemWithCutBase<OffPho>*>& histVec,const std::string& filterName,const std::string& baseName,int nrBins,double xMin,double xMax,float (OffPho::*vsVarFunc)()const,const CutMasks& masks);   
    void initStdEffHists(std::vector<MonElemWithCutBase<OffPho>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData::Data1D& bins,float (OffPho::*vsVarFunc)()const,const CutMasks& masks);

    //we own the passed in pointer
    void initStdEleCutHists(std::vector<MonElemWithCutBase<OffEle>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData& bins,EgHLTDQMCut<OffEle>* cut=NULL);
    void initStdPhoCutHists(std::vector<MonElemWithCutBase<OffPho>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData& bins,EgHLTDQMCut<OffPho>* cut=NULL);


    
  
    void initTightLooseTrigHists( std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins,EgHLTDQMCut<OffEle>* eleCut);
    void initTightLooseTrigHists( std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins,EgHLTDQMCut<OffPho>* phoCut);
    
    void initTightLooseTrigHistsTrigCuts( std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins);  
    void initTightLooseTrigHistsTrigCuts( std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins);
    
    void addTightLooseTrigHist( std::vector<MonElemContainer<OffEle>*>& eleMonElems,
			       const std::string& tightTrig,const std::string& looseTrig,
			       EgHLTDQMCut<OffEle>* eleCut,const std::string& histId,const BinData& bins);


    void addTightLooseTrigHist( std::vector<MonElemContainer<OffPho>*>& phoMonElems,
			       const std::string& tightTrig,const std::string& looseTrig,
			       EgHLTDQMCut<OffPho>* phoCut,const std::string& histId,const BinData& bins);
    

    
    void initTightLooseDiObjTrigHistsTrigCuts( std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins);
    void initTightLooseDiObjTrigHistsTrigCuts( std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins);

    //ele only (Now for pho also!)
    void initTrigTagProbeHists(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string> filterNames,int cutMask,const BinData& bins);
    void initTrigTagProbeHists(std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::vector<std::string> filterNames,int cutMask,const BinData& bins);
    void initTrigTagProbeHist(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::string filterName,int cutMask,const BinData& bins);
    void initTrigTagProbeHist(std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::string filterName,int cutMask,const BinData& bins);
    void initTrigTagProbeHist_2Leg(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::string filterName,int cutMask,const BinData& bins);
  

    template<class T,typename varType> void addStdHist( std::vector<MonElemManagerBase<T>*>& histVec,const std::string& name,const std::string& title,
						       const BinData::Data1D& binData,varType (T::*varFunc)()const){
      histVec.push_back(new MonElemMgrEBEE<T,varType>(iBooker, name,title,binData.nr,binData.min,binData.max,varFunc));
    }

    //this function is special in that it figures out the Et cut from the trigger name
    //it then passes the cut as normal into the other addTightLooseTrigHist functions
    //it also makes an uncut et distribution
    template<class T> void addTightLooseTrigHist( std::vector<MonElemContainer<T>*>& monElems,
					 const std::string& tightTrig,const std::string& looseTrig,
					 const std::string& histId,const BinData& bins)
    {
  
      float etCutValue = trigTools::getEtThresFromName(tightTrig);
      
      EgHLTDQMCut<T>* etCut = new EgGreaterCut<T,float>(etCutValue,&T::etSC); //note the cut in trigger is on SC Et
      addTightLooseTrigHist(monElems,tightTrig,looseTrig,etCut,histId,bins);

      //now make the new mon elems without the et cut (have to be placed in containers even though each container just has one monelem)
      MonElemContainer<T>* passEtMonElem;
      passEtMonElem = new MonElemContainer<T>(tightTrig+"_"+looseTrig+"_"+histId+"_passTrig","",
					      new EgObjTrigCut<T>(trigCodes.getCode(tightTrig+":"+looseTrig),EgObjTrigCut<T>::AND));
      addStdHist<T,float>(passEtMonElem->monElems(),passEtMonElem->name()+"_etUnCut",passEtMonElem->name()+" E_{T} (Uncut);E_{T} (GeV)",bins.et,&T::et);
      
      MonElemContainer<T>* failEtMonElem;
      failEtMonElem = new MonElemContainer<T>(tightTrig+"_"+looseTrig+"_"+histId+"_failTrig","",
					      new EgObjTrigCut<T>(trigCodes.getCode(looseTrig),EgObjTrigCut<T>::AND,trigCodes.getCode(tightTrig)));
      addStdHist<T,float>(failEtMonElem->monElems(),failEtMonElem->name()+"_etUnCut",failEtMonElem->name()+" E_{T} (Uncut);E_{T} (GeV)",bins.et,&T::et);

      monElems.push_back(passEtMonElem);
      monElems.push_back(failEtMonElem);
    }
  

    //this function will ultimately produce a set of distributions with the Et cut of the trigger applied + make an additional un cut et monitor element for turn on purposes
    template<class T> void initTightLooseTrigHists(std::vector<MonElemContainer<T>*>& monElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins,const std::string& objName)
      {
	for(size_t trigNr=0;trigNr<tightLooseTrigs.size();trigNr++){
	  //dbe_->SetCurrentFolder(dirName_+"/"+tightLooseTrigs[trigNr]);
	  std::vector<std::string> splitString;
	  boost::split(splitString,tightLooseTrigs[trigNr],boost::is_any_of(std::string(":")));
	  if(splitString.size()!=2) continue; //format incorrect
	  const std::string& tightTrig = splitString[0];
	  const std::string& looseTrig = splitString[1];
	  //this step is necessary as we want to transfer ownership of eleCut to the addTrigLooseTrigHist func on the last iteration
	  //but clone it before that
	  //perhaps my object ownership rules need to be re-evalulated
	  if(trigNr!=tightLooseTrigs.size()-2) addTightLooseTrigHist(monElems,tightTrig,looseTrig,objName,bins);
	  else addTightLooseTrigHist(monElems,tightTrig,looseTrig,objName,bins);
	}
	//dbe_->SetCurrentFolder(dirName_);
      }
    

  }; // end of class
}
#endif
