#ifndef DQMOffline_Trigger_HLTTagAndProbeEff_h
#define DQMOffline_Trigger_HLTTagAndProbeEff_h


#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/RegexMatch.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <boost/algorithm/string.hpp>

#include <vector>
#include <string>

//functions we wish to add that are not direct member functions
namespace{
  template<typename ObjType> float scEtaFunc(const ObjType& obj){return obj.superCluster()->eta();}
 
  template<typename ObjType>
  std::function<float(const ObjType&)> getFunc(const std::string& varName){  
    std::function<float(const ObjType&)> varFunc;
    if(varName=="et") varFunc = &ObjType::et;
    else if(varName=="pt") varFunc = &ObjType::pt;
    else if(varName=="eta") varFunc = &ObjType::eta;
    else if(varName=="phi") varFunc = &ObjType::phi;
    else if(varName=="scEta") varFunc = scEtaFunc<ObjType>;
    else if(!varName.empty()){
      throw cms::Exception("ConfigError") <<"var "<<varName<<" not recognised"<<std::endl;
    }
    return varFunc;
  }
}

namespace{
  bool passTrig(const float objEta,float objPhi, const trigger::TriggerEvent& trigEvt,
		const std::string & filterName,const std::string& processName){
    constexpr float kMaxDR2 = 0.1*0.1;
    
    edm::InputTag filterTag(filterName,"",processName); 
    trigger::size_type filterIndex = trigEvt.filterIndex(filterTag); 
    if(filterIndex<trigEvt.sizeFilters()){ //check that filter is in triggerEvent
      const trigger::Keys& trigKeys = trigEvt.filterKeys(filterIndex); 
      const trigger::TriggerObjectCollection & trigObjColl(trigEvt.getObjects());
      for(trigger::Keys::const_iterator keyIt=trigKeys.begin();keyIt!=trigKeys.end();++keyIt){ 
	const trigger::TriggerObject& trigObj = trigObjColl[*keyIt];
	if(reco::deltaR2(trigObj.eta(),trigObj.phi(),objEta,objPhi)<kMaxDR2) return true;
      }
    }
    return false;
  }
  bool passTrig(const float objEta,float objPhi, const trigger::TriggerEvent& trigEvt,
		const std::vector<std::string>& filterNames,const std::string& processName){
    for(auto& filterName : filterNames){
      if(passTrig(objEta,objPhi,trigEvt,filterName,processName)==false) return false;
    }
    return true;
  }
   //inspired by https://github.com/cms-sw/cmssw/blob/fc4f8bbe1258790e46e2d554aacea15c3e5d9afa/HLTrigger/HLTfilters/src/HLTHighLevel.cc#L124-L165
   //triggers are ORed together
   //empty pattern is auto pass
  bool passTrig(const std::string& trigPattern,const edm::TriggerNames& trigNames,const edm::TriggerResults& trigResults){
    
    if(trigPattern.empty()) return true;
    
    std::vector<std::string> trigNamesToPass;
    if(edm::is_glob(trigPattern)){
     
      //matches is vector of string iterators
      const auto& matches = edm::regexMatch(trigNames.triggerNames(),trigPattern);
      for(auto& name : matches) trigNamesToPass.push_back(*name);
    }else{
      trigNamesToPass.push_back(trigPattern); //not a pattern, much be a path
    }
    for(auto& trigName : trigNamesToPass){
      size_t pathIndex = trigNames.triggerIndex(trigName);
      if(pathIndex<trigResults.size() && 
	 trigResults.accept(pathIndex)) return true; 
    }
    
    return false;
  }
  
  template<typename T> edm::Handle<T> getHandle(const edm::Event& event,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    event.getByToken(token,handle);
    return handle;
  }
}


template<typename ObjType>
class RangeCuts {
public:
  RangeCuts(const edm::ParameterSet& config){
    varName_ = config.getParameter<std::string>("rangeVar");
    varFunc_ = getFunc<ObjType>(varName_);
    auto ranges = config.getParameter<std::vector<std::string> >("allowedRanges");
    for(auto range: ranges){
      std::vector<std::string> splitRange;
      boost::split(splitRange,range,boost::is_any_of(":"));
      if(splitRange.size()!=2) throw cms::Exception("ConfigError") <<"range "<<range<<" is not of format X:Y"<<std::endl;
      allowedRanges_.push_back({std::stof(splitRange[0]),std::stof(splitRange[1])});
    }
  }
  bool operator()(const ObjType& obj)const{
    if(!varFunc_) return true; //auto pass if we dont specify a variable function
    else{ 
      float varVal = varFunc_(obj);
      for(auto& range : allowedRanges_){
	if(varVal>=range.first && varVal<range.second) return true;
      }
      return false;
    }
  }
  const std::string& varName()const{return varName_;}
  private:
  std::string varName_;
  std::function<float(const ObjType&)> varFunc_;
  std::vector<std::pair<float,float> > allowedRanges_;
};

template<typename ObjType>
class RangeCutsColl {
public:
  explicit RangeCutsColl(const std::vector<edm::ParameterSet>& configs){
    for(const auto & cutConfig : configs) rangeCuts_.emplace_back(RangeCuts<ObjType>(cutConfig));
  }
  //if no cuts are defined, it returns true
  bool operator()(const ObjType& obj)const{
    for(auto& cut : rangeCuts_){
      if(!cut(obj)) return false;
    }
    return true;
  }
  //if no cuts are defined, it returns true
  //this version allows us to skip a range check for a specificed variable
  //okay this feature requirement was missed in the initial (very rushed) design phase
  //and thats why its now hacked in 
  //basically if you're applying an Et cut, you want to automatically turn it of
  //when you're making a turn on curve...
  bool operator()(const ObjType& obj,const std::string& varToSkip)const{
    for(auto& cut : rangeCuts_){
      if(cut.varName()==varToSkip) continue;
      if(!cut(obj)) return false;
    }
    return true;
  }
  bool operator()(const ObjType& obj,const std::vector<std::string>& varsToSkip)const{
    for(auto& cut : rangeCuts_){
      if(std::find(varsToSkip.begin(),varsToSkip.end(),cut.varName())!=varsToSkip.end()) continue;
      if(!cut(obj)) return false;
    }
    return true;
  }
private:
  std::vector<RangeCuts<ObjType> > rangeCuts_;
};

//our base class for our histograms
//takes an object, edm::Event,edm::EventSetup and fills the histogram
//with the predetermined variable (or varaibles) 
template <typename ObjType> 
class HLTDQMHist {
public:
  HLTDQMHist()=default;
  virtual ~HLTDQMHist()=default;
  virtual void fill(const ObjType& objType,const edm::Event& event,
		    const edm::EventSetup& setup,const RangeCutsColl<ObjType>& rangeCuts)=0;
};


//this class is a specific implimentation of a HLTDQMHist
//it has the value with which to fill the histogram 
//and the histogram itself
//we do not own the histogram
template <typename ObjType,typename ValType> 
class HLTDQM1DHist : public HLTDQMHist<ObjType> {
public:
  HLTDQM1DHist(TH1* hist,const std::string& varName,
	       std::function<ValType(const ObjType&)> func,
	       const RangeCutsColl<ObjType>& rangeCuts):
    var_(func),varName_(varName),localRangeCuts_(rangeCuts),hist_(hist){}
  void fill(const ObjType& obj,const edm::Event& event,
	    const edm::EventSetup& setup,const RangeCutsColl<ObjType>& globalRangeCuts)override{
    if(globalRangeCuts(obj,varName_) && localRangeCuts_(obj)){  //local range cuts are specific to a histogram so dont ignore variables like global ones (all local cuts should be approprate)
      hist_->Fill(var_(obj));
    }
  }
private:
  std::function<ValType(const ObjType&)> var_;
  std::string varName_;
  RangeCutsColl<ObjType> localRangeCuts_;
  TH1* hist_; //we do not own this
};

template <typename ObjType,typename XValType,typename YValType=XValType> 
class HLTDQM2DHist : public HLTDQMHist<ObjType> {
public:
  HLTDQM2DHist(TH2* hist,const std::string& xVarName,const std::string& yVarName,
	       std::function<XValType(const ObjType&)> xFunc,
	       std::function<YValType(const ObjType&)> yFunc,
	       const RangeCutsColl<ObjType>& rangeCuts):
    xVar_(xFunc),yVar_(yFunc),
    xVarName_(xVarName),yVarName_(yVarName),
    localRangeCuts_(rangeCuts),hist_(hist){}

  void fill(const ObjType& obj,const edm::Event& event,
	    const edm::EventSetup& setup,const RangeCutsColl<ObjType>& globalRangeCuts)override{
    if(globalRangeCuts(obj,std::vector<std::string>{xVarName_,yVarName_}) &&
       localRangeCuts_(obj)){  //local range cuts are specific to a histogram so dont ignore variables like global ones (all local cuts should be approprate)
      hist_->Fill(xVar_(obj),yVar_(obj));
    }
  }
private:
  std::function<XValType(const ObjType&)> xVar_;
  std::function<YValType(const ObjType&)> yVar_;
  std::string xVarName_;
  std::string yVarName_;
  RangeCutsColl<ObjType> localRangeCuts_;
  TH2* hist_; //we do not own this
};

template <typename ObjType> 
class HLTDQMHistColl {
public:
  
  explicit HLTDQMHistColl(const edm::ParameterSet& config,
			  const std::string& baseHistName,
			  const std::string& hltProcess);
  void bookHists(DQMStore::IBooker& iBooker,const std::vector<edm::ParameterSet>& histConfigs);
  void fillHists(const ObjType& tag,const ObjType& probe,const edm::Event& event,
		 const edm::EventSetup& setup,const trigger::TriggerEvent& trigEvt);
private:
  void book1D(DQMStore::IBooker& iBooker,const edm::ParameterSet& histConfig);
  void book2D(DQMStore::IBooker& iBooker,const edm::ParameterSet& histConfig);
private:
  std::vector<std::unique_ptr<HLTDQMHist<ObjType> > > histsPass_;
  std::vector<std::unique_ptr<HLTDQMHist<ObjType> > > histsTot_;
  RangeCutsColl<ObjType> rangeCuts_;
  std::string filterName_;
  std::string tagExtraFilter_;
  std::string baseHistName_;
  std::string histTitle_;
  std::string folderName_;
  std::string hltProcess_;

};

template <typename ObjType> 
HLTDQMHistColl<ObjType>::HLTDQMHistColl(const edm::ParameterSet& config,
					const std::string& baseHistName,
					const std::string& hltProcess):
  rangeCuts_(config.getParameter<std::vector<edm::ParameterSet> >("rangeCuts")),
  filterName_(config.getParameter<std::string>("filterName")), 
  tagExtraFilter_(config.getParameter<std::string>("tagExtraFilter")), 
  baseHistName_(baseHistName),
  histTitle_(config.getParameter<std::string>("histTitle")),
  folderName_(config.getParameter<std::string>("folderName")),
  hltProcess_(hltProcess)
{
  
}

template <typename ObjType>
void HLTDQMHistColl<ObjType>::bookHists(DQMStore::IBooker& iBooker,const std::vector<edm::ParameterSet>& histConfigs)
{
  iBooker.setCurrentFolder(folderName_);
  for(auto& histConfig : histConfigs){
    std::string histType = histConfig.getParameter<std::string>("histType");
    if(histType=="1D"){
      book1D(iBooker,histConfig);
    }else if(histType=="2D"){
      book2D(iBooker,histConfig);
    }else{
      throw cms::Exception("ConfigError")<<" histType "<<histType<<" not recognised"<<std::endl;
    }
  }
}

template <typename ObjType>
void HLTDQMHistColl<ObjType>::book1D(DQMStore::IBooker& iBooker,const edm::ParameterSet& histConfig)
{
  auto binLowEdgesDouble = histConfig.getParameter<std::vector<double> >("binLowEdges");
  std::vector<float> binLowEdges;
  for(double lowEdge : binLowEdgesDouble) binLowEdges.push_back(lowEdge);
  auto nameSuffex = histConfig.getParameter<std::string>("nameSuffex");
  auto mePass = iBooker.book1D((baseHistName_+filterName_+nameSuffex+"_pass").c_str(),
			       (histTitle_+nameSuffex+" Pass").c_str(),
			       binLowEdges.size()-1,&binLowEdges[0]);
  std::unique_ptr<HLTDQMHist<ObjType> > hist;
  auto vsVar = histConfig.getParameter<std::string>("vsVar");
  auto vsVarFunc = getFunc<ObjType>(vsVar);
  if(!vsVarFunc) {
    throw cms::Exception("ConfigError")<<" vsVar "<<vsVar<<" is giving null ptr (likely empty)"<<std::endl;
  }
  RangeCutsColl<ObjType> rangeCuts(histConfig.getParameter<std::vector<edm::ParameterSet> >("rangeCuts"));
  hist = std::make_unique<HLTDQM1DHist<ObjType,float> >(mePass->getTH1(),vsVar,vsVarFunc,rangeCuts); 
  histsPass_.emplace_back(std::move(hist));
  auto meTot = iBooker.book1D((baseHistName_+filterName_+nameSuffex+"_tot").c_str(),
			      (histTitle_+nameSuffex+" Total").c_str(),
			      binLowEdges.size()-1,&binLowEdges[0]);
  hist = std::make_unique<HLTDQM1DHist<ObjType,float> >(meTot->getTH1(),vsVar,vsVarFunc,rangeCuts); 
  histsTot_.emplace_back(std::move(hist));
  
}

template <typename ObjType>
void HLTDQMHistColl<ObjType>::book2D(DQMStore::IBooker& iBooker,const edm::ParameterSet& histConfig)
{
  auto xBinLowEdgesDouble = histConfig.getParameter<std::vector<double> >("xBinLowEdges");
  auto yBinLowEdgesDouble = histConfig.getParameter<std::vector<double> >("yBinLowEdges");
  std::vector<float> xBinLowEdges;
  std::vector<float> yBinLowEdges;
  for(double lowEdge : xBinLowEdgesDouble) xBinLowEdges.push_back(lowEdge);
  for(double lowEdge : yBinLowEdgesDouble) yBinLowEdges.push_back(lowEdge);
  auto nameSuffex = histConfig.getParameter<std::string>("nameSuffex");
  auto mePass = iBooker.book2D((baseHistName_+filterName_+nameSuffex+"_pass").c_str(),
			       (histTitle_+nameSuffex+" Pass").c_str(),
 			       xBinLowEdges.size()-1,&xBinLowEdges[0],
			       yBinLowEdges.size()-1,&yBinLowEdges[0]);
  std::unique_ptr<HLTDQMHist<ObjType> > hist;
  auto xVar = histConfig.getParameter<std::string>("xVar");
  auto yVar = histConfig.getParameter<std::string>("yVar");
  auto xVarFunc = getFunc<ObjType>(xVar);
  auto yVarFunc = getFunc<ObjType>(yVar);
  if(!xVarFunc || !yVarFunc) {
    throw cms::Exception("ConfigError")<<" xVar "<<xVar<<" or yVar "<<yVar<<" is giving null ptr (likely empty)"<<std::endl;
  }
  RangeCutsColl<ObjType> rangeCuts(histConfig.getParameter<std::vector<edm::ParameterSet> >("rangeCuts"));
  

  //really? really no MonitorElement::getTH2...sigh
  hist = std::make_unique<HLTDQM2DHist<ObjType,float> >(static_cast<TH2*>(mePass->getTH1()),xVar,yVar,xVarFunc,yVarFunc,rangeCuts); 
  histsPass_.emplace_back(std::move(hist));
  
  auto meTot = iBooker.book2D((baseHistName_+filterName_+nameSuffex+"_tot").c_str(),
			      (histTitle_+nameSuffex+" Total").c_str(),
			      xBinLowEdges.size()-1,&xBinLowEdges[0],
			      yBinLowEdges.size()-1,&yBinLowEdges[0]);

  hist = std::make_unique<HLTDQM2DHist<ObjType,float> >(static_cast<TH2*>(meTot->getTH1()),xVar,yVar,xVarFunc,yVarFunc,rangeCuts); 
  
  histsTot_.emplace_back(std::move(hist));
}

template <typename ObjType>
void HLTDQMHistColl<ObjType>::fillHists(const ObjType& tag,
					const ObjType& probe,
					const edm::Event& event,
					const edm::EventSetup& setup,
					const trigger::TriggerEvent& trigEvt)
{
  if(tagExtraFilter_.empty() || passTrig(tag.eta(),tag.phi(),trigEvt,tagExtraFilter_,hltProcess_)){
    for(auto& hist : histsTot_){
      hist->fill(probe,event,setup,rangeCuts_); 
    }
    
    if(passTrig(probe.eta(),probe.phi(),trigEvt,filterName_,hltProcess_)){
      for(auto& hist : histsPass_){
	hist->fill(probe,event,setup,rangeCuts_); 
      }
      
    }
  }
}


template <typename ObjType,typename ObjCollType> 
class HLTTagAndProbeEff {
public:
    
  explicit HLTTagAndProbeEff(const edm::ParameterSet& pset,edm::ConsumesCollector && cc);
 
  void bookHists(DQMStore::IBooker& iBooker);
  void fill(const edm::Event& event,const edm::EventSetup& setup);
  
private:
  std::vector<edm::Ref<ObjCollType> >
  getPassingRefs(const edm::Handle<ObjCollType>& objCollHandle,
		 const trigger::TriggerEvent& trigEvt,
		 const std::vector<std::string>& filterNames,
		 const edm::Handle<edm::ValueMap<bool> >& vidHandle,
		 const RangeCutsColl<ObjType>& rangeCuts);
private:
  edm::EDGetTokenT<ObjCollType> objToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> trigEvtToken_;
  edm::EDGetTokenT<edm::TriggerResults> trigResultsToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> >tagVIDToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> >probeVIDToken_;
  
  std::string hltProcess_;

  std::string tagTrigger_;
	    
  std::vector<std::string> tagFilters_;
  RangeCutsColl<ObjType> tagRangeCuts_;

  std::vector<std::string> probeFilters_;
  RangeCutsColl<ObjType> probeRangeCuts_;

  float minMass_;
  float maxMass_;
  bool requireOpSign_;

  std::vector<edm::ParameterSet> histConfigs_;
  std::vector<HLTDQMHistColl<ObjType> > histColls_;
    
};

template <typename ObjType,typename ObjCollType> 
HLTTagAndProbeEff<ObjType,ObjCollType>::HLTTagAndProbeEff(const edm::ParameterSet& pset,edm::ConsumesCollector && cc):
  tagRangeCuts_(pset.getParameter<std::vector<edm::ParameterSet> >("tagRangeCuts")),
  probeRangeCuts_(pset.getParameter<std::vector<edm::ParameterSet> >("probeRangeCuts"))
{
  edm::InputTag trigEvtTag = pset.getParameter<edm::InputTag>("trigEvent");

  objToken_ = cc.consumes<ObjCollType>(pset.getParameter<edm::InputTag>("objColl"));
  trigEvtToken_ = cc.consumes<trigger::TriggerEvent>(trigEvtTag);
  trigResultsToken_ = cc.consumes<edm::TriggerResults>(pset.getParameter<edm::InputTag>("trigResults"));
  tagVIDToken_ = cc.consumes<edm::ValueMap<bool> >(pset.getParameter<edm::InputTag>("tagVIDCuts"));
  probeVIDToken_ = cc.consumes<edm::ValueMap<bool> >(pset.getParameter<edm::InputTag>("probeVIDCuts"));

  hltProcess_ = trigEvtTag.process();

  tagTrigger_ = pset.getParameter<std::string>("tagTrigger");

  tagFilters_ = pset.getParameter<std::vector<std::string> >("tagFilters");
  probeFilters_ = pset.getParameter<std::vector<std::string> >("probeFilters");
  
  minMass_ = pset.getParameter<double>("minMass");
  maxMass_ = pset.getParameter<double>("maxMass");
  requireOpSign_ = pset.getParameter<bool>("requireOpSign");

  histConfigs_ = pset.getParameter<std::vector<edm::ParameterSet> >("histConfigs");
  const auto& histCollConfigs = pset.getParameter<std::vector<edm::ParameterSet> >("histCollConfigs");
  
  std::string baseHistName = pset.getParameter<std::string>("baseHistName");

  for(auto& config: histCollConfigs){
    histColls_.emplace_back(HLTDQMHistColl<ObjType>(config,baseHistName,hltProcess_));
  }

}

template <typename ObjType,typename ObjCollType> 
std::vector<edm::Ref<ObjCollType> > HLTTagAndProbeEff<ObjType,ObjCollType>::
getPassingRefs(const edm::Handle<ObjCollType>& objCollHandle,
	       const trigger::TriggerEvent& trigEvt,
	       const std::vector<std::string>& filterNames,
	       const edm::Handle<edm::ValueMap<bool> >& vidHandle,
	       const RangeCutsColl<ObjType>& rangeCuts)
{
  std::vector<edm::Ref<ObjCollType> > passingRefs;
  for(size_t objNr=0;objNr<objCollHandle->size();objNr++){
    edm::Ref<ObjCollType> ref(objCollHandle,objNr);
    if(rangeCuts(*ref) && 
       passTrig(ref->eta(),ref->phi(),trigEvt,filterNames,hltProcess_) && 
       (vidHandle.isValid()==false || (*vidHandle)[ref]==true)){
      passingRefs.push_back(ref);
    }
  }
  return passingRefs;
}


template <typename ObjType,typename ObjCollType> 
void HLTTagAndProbeEff<ObjType,ObjCollType>::fill(const edm::Event& event,const edm::EventSetup& setup)
{
  auto objCollHandle = getHandle(event,objToken_); 
  auto trigEvtHandle = getHandle(event,trigEvtToken_);
  auto trigResultsHandle = getHandle(event,trigResultsToken_);
  auto tagVIDHandle = getHandle(event,tagVIDToken_);
  auto probeVIDHandle = getHandle(event,probeVIDToken_);

  //we need the object collection and trigger info at the minimum
  if(!objCollHandle.isValid() || !trigEvtHandle.isValid() || !trigResultsHandle.isValid()) return;

  //now check if the tag trigger fired, return if not
  //if no trigger specified it passes
  const edm::TriggerNames& trigNames = event.triggerNames(*trigResultsHandle);
  if(passTrig(tagTrigger_,trigNames,*trigResultsHandle)==false) return;
  
  std::vector<edm::Ref<ObjCollType> > tagRefs = getPassingRefs(objCollHandle,*trigEvtHandle,
							       tagFilters_,
							       tagVIDHandle,tagRangeCuts_);

  std::vector<edm::Ref<ObjCollType> > probeRefs = getPassingRefs(objCollHandle,*trigEvtHandle,
								 probeFilters_,
								 probeVIDHandle,probeRangeCuts_);

  for(auto& tagRef : tagRefs){
    for(auto& probeRef : probeRefs){
      if(tagRef==probeRef) continue; //otherwise its the same object...
      float mass = (tagRef->p4()+probeRef->p4()).mag();
      if( ( mass>minMass_ || minMass_<0 ) && 
	  ( mass<maxMass_ || maxMass_<0 ) && 
	  ( !requireOpSign_ || tagRef->charge()!=probeRef->charge()) ){
	for(auto& histColl : histColls_){
	  histColl.fillHists(*tagRef,*probeRef,event,setup,*trigEvtHandle);
	}	      
      }//end of t&p pair cuts
    }//end of probe loop
  }//end of tag loop
}

template <typename ObjType,typename ObjCollType> 
void HLTTagAndProbeEff<ObjType,ObjCollType>::bookHists(DQMStore::IBooker& iBooker)
{
  for(auto& histColl:  histColls_) histColl.bookHists(iBooker,histConfigs_);
}

#endif
