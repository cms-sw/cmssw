#ifndef DQMOffline_Trigger_HLTTagAndProbeEff_h
#define DQMOffline_Trigger_HLTTagAndProbeEff_h

//***************************************************************************
//
// Description: 
//   This class produces histograms which can be used to make tag&probe efficiencies
//   for specified HLT filters 
//   These histograms can be binned in various variables
//   
//   In a nutshell: 
//   1) requires the tag trigger to pass for the event
//   2) creates a collection of tags, with the tags required to pass the a specified
//      filter and selection ID. The selection is in the form of a value map calculated
//      by a previous module. Additional kinematic cuts can be applied (its not limited to 
//      kinematic cuts per se but thats the intention) 
//   3) likewise creates a collection of probes passing filter, ID and kinematic requirements
//   4) applies selection like mass & opp sign to the t&p pair
//   5) passes the t&p pair to HLTDQMFilterTnPEffHists, with each one corresponding to a 
//      specific HLT filter whose efficiency we wish to measure. This class then creates the 
//      numerator and denominator histograms for the efficiency calculation done at harvesting 
//
//
// Author: Sam Harper (RAL) , 2017
//
//***************************************************************************



#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DQMOffline/Trigger/interface/UtilFuncs.h"
#include "DQMOffline/Trigger/interface/VarRangeCutColl.h"
#include "DQMOffline/Trigger/interface/HLTDQMFilterTnPEffHists.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include <vector>
#include <string>

namespace{
 template<typename T> edm::Handle<T> getHandle(const edm::Event& event,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    event.getByToken(token,handle);
    return handle;
  }
}

template <typename ObjType,typename ObjCollType> 
class HLTDQMTagAndProbeEff {
public:
    
  explicit HLTDQMTagAndProbeEff(const edm::ParameterSet& pset,edm::ConsumesCollector && cc);
  
  static edm::ParameterSetDescription makePSetDescription();
  
  void beginRun(const edm::Run& run,const edm::EventSetup& setup);
  void bookHists(DQMStore::IBooker& iBooker);
  void fill(const edm::Event& event,const edm::EventSetup& setup);
  
private:
  std::vector<edm::Ref<ObjCollType> >
  getPassingRefs(const edm::Handle<ObjCollType>& objCollHandle,
		 const trigger::TriggerEvent& trigEvt,
		 const std::vector<std::string>& filterNames, 
		 const bool orFilters,
		 const edm::Handle<edm::ValueMap<bool> >& vidHandle,
		 const VarRangeCutColl<ObjType>& rangeCuts);
private:
  edm::EDGetTokenT<ObjCollType> objToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> trigEvtToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> >tagVIDToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> >probeVIDToken_;
  
  std::string hltProcess_;

  std::string tagTrigger_;
	    
  std::vector<std::string> tagFilters_;
  bool tagFiltersORed_;//true=ORed, false=ANDed
  VarRangeCutColl<ObjType> tagRangeCuts_;

  std::vector<std::string> probeFilters_;
  bool probeFiltersORed_;//true=ORed, false=ANDed
  VarRangeCutColl<ObjType> probeRangeCuts_;

  float minMass_;
  float maxMass_;
  bool requireOpSign_;

  std::vector<edm::ParameterSet> histConfigs_;
  std::vector<HLTDQMFilterTnPEffHists<ObjType> > filterHists_;
  
  GenericTriggerEventFlag sampleTrigRequirements_;

};

template <typename ObjType,typename ObjCollType> 
HLTDQMTagAndProbeEff<ObjType,ObjCollType>::HLTDQMTagAndProbeEff(const edm::ParameterSet& pset,edm::ConsumesCollector && cc):
  tagRangeCuts_(pset.getParameter<std::vector<edm::ParameterSet> >("tagRangeCuts")),
  probeRangeCuts_(pset.getParameter<std::vector<edm::ParameterSet> >("probeRangeCuts")),
  sampleTrigRequirements_(pset.getParameter<edm::ParameterSet>("sampleTrigRequirements"),cc)
{
  edm::InputTag trigEvtTag = pset.getParameter<edm::InputTag>("trigEvent");

  objToken_ = cc.consumes<ObjCollType>(pset.getParameter<edm::InputTag>("objColl"));
  trigEvtToken_ = cc.consumes<trigger::TriggerEvent>(trigEvtTag);
  tagVIDToken_ = cc.consumes<edm::ValueMap<bool> >(pset.getParameter<edm::InputTag>("tagVIDCuts"));
  probeVIDToken_ = cc.consumes<edm::ValueMap<bool> >(pset.getParameter<edm::InputTag>("probeVIDCuts"));

  hltProcess_ = trigEvtTag.process();

  tagFilters_ = pset.getParameter<std::vector<std::string> >("tagFilters");
  tagFiltersORed_ = pset.getParameter<bool>("tagFiltersORed");
  probeFilters_ = pset.getParameter<std::vector<std::string> >("probeFilters");
  probeFiltersORed_ = pset.getParameter<bool>("tagFiltersORed");
  
  
  minMass_ = pset.getParameter<double>("minMass");
  maxMass_ = pset.getParameter<double>("maxMass");
  requireOpSign_ = pset.getParameter<bool>("requireOpSign");

  histConfigs_ = pset.getParameter<std::vector<edm::ParameterSet> >("histConfigs");
  const auto& filterConfigs = pset.getParameter<std::vector<edm::ParameterSet> >("filterConfigs");
  
  std::string baseHistName = pset.getParameter<std::string>("baseHistName");

  for(auto& config: filterConfigs){
    filterHists_.emplace_back(HLTDQMFilterTnPEffHists<ObjType>(config,baseHistName,hltProcess_));
  }
}

template <typename ObjType,typename ObjCollType> 
edm::ParameterSetDescription HLTDQMTagAndProbeEff<ObjType,ObjCollType>::
makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.addVPSet("tagRangeCuts",VarRangeCut<ObjType>::makePSetDescription(),std::vector<edm::ParameterSet>());
  desc.addVPSet("probeRangeCuts",VarRangeCut<ObjType>::makePSetDescription(),std::vector<edm::ParameterSet>());
  desc.add<edm::InputTag>("trigEvent",edm::InputTag("hltTriggerSummaryAOD","","HLT"));
  desc.add<edm::InputTag>("objColl",edm::InputTag());
  desc.add<edm::InputTag>("tagVIDCuts",edm::InputTag());
  desc.add<edm::InputTag>("probeVIDCuts",edm::InputTag());
  desc.add<std::vector<std::string> >("tagFilters",std::vector<std::string>());
  desc.add<std::vector<std::string> >("probeFilters",std::vector<std::string>());
  desc.add<bool>("tagFiltersORed",true);//default to OR probe filters (use case is multiple tag triggers, eg Ele27, Ele32, Ele35 tight etc)
  desc.add<bool>("probeFiltersORed",false); //default to AND probe filters (cant think why you would want to OR them but made if configurable just in case)
  desc.add<double>("minMass");
  desc.add<double>("maxMass");
  desc.add<bool>("requireOpSign");
  desc.addVPSet("histConfigs",HLTDQMFilterTnPEffHists<ObjType>::makePSetDescriptionHistConfigs(),std::vector<edm::ParameterSet>()); 
  desc.addVPSet("filterConfigs",HLTDQMFilterTnPEffHists<ObjType>::makePSetDescription(),std::vector<edm::ParameterSet>()); 
  desc.add<std::string>("baseHistName");

  edm::ParameterSetDescription trigEvtFlagDesc;
  trigEvtFlagDesc.add<bool>("andOr",false);
  trigEvtFlagDesc.add<unsigned int>("verbosityLevel",1);
  trigEvtFlagDesc.add<bool>("andOrDcs", false);  
  trigEvtFlagDesc.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi") );
  trigEvtFlagDesc.add<std::vector<int> >("dcsPartitions",{24,25,26,27,28,29});
  trigEvtFlagDesc.add<bool>("errorReplyDcs", true);
  trigEvtFlagDesc.add<std::string>("dbLabel","");
  trigEvtFlagDesc.add<bool>("andOrHlt", true); //true = OR, false = and
  trigEvtFlagDesc.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT") );
  trigEvtFlagDesc.add<std::vector<std::string> >("hltPaths",{});
  trigEvtFlagDesc.add<std::string>("hltDBKey","");
  trigEvtFlagDesc.add<bool>("errorReplyHlt",false);
  desc.add<edm::ParameterSetDescription>("sampleTrigRequirements",trigEvtFlagDesc);
  
  return desc;
}

template <typename ObjType,typename ObjCollType> 
void HLTDQMTagAndProbeEff<ObjType,ObjCollType>::bookHists(DQMStore::IBooker& iBooker)
{
  for(auto& filterHist:  filterHists_) filterHist.bookHists(iBooker,histConfigs_);
}

template <typename ObjType,typename ObjCollType> 
void HLTDQMTagAndProbeEff<ObjType,ObjCollType>::
beginRun(const edm::Run& run,const edm::EventSetup& setup)
{
  if(sampleTrigRequirements_.on()) sampleTrigRequirements_.initRun(run,setup);
}


template <typename ObjType,typename ObjCollType> 
void HLTDQMTagAndProbeEff<ObjType,ObjCollType>::fill(const edm::Event& event,const edm::EventSetup& setup)
{
  auto objCollHandle = getHandle(event,objToken_); 
  auto trigEvtHandle = getHandle(event,trigEvtToken_);
  auto tagVIDHandle = getHandle(event,tagVIDToken_);
  auto probeVIDHandle = getHandle(event,probeVIDToken_);

  //we need the object collection and trigger info at the minimum
  if(!objCollHandle.isValid() || !trigEvtHandle.isValid()) return;

  //if GenericTriggerEventFlag is "off", it'll return true regardless
  //if so if its off, we auto pass which is the behaviour we wish to have
  //if its null, we auto fail (because that shouldnt happen)
  if(sampleTrigRequirements_.accept(event,setup)==false) return;
  
  std::vector<edm::Ref<ObjCollType> > tagRefs = getPassingRefs(objCollHandle,*trigEvtHandle,
							       tagFilters_,tagFiltersORed_,
							       tagVIDHandle,tagRangeCuts_);

  std::vector<edm::Ref<ObjCollType> > probeRefs = getPassingRefs(objCollHandle,*trigEvtHandle,
								 probeFilters_,probeFiltersORed_,
								 probeVIDHandle,probeRangeCuts_);

  for(auto& tagRef : tagRefs){
    for(auto& probeRef : probeRefs){
      if(tagRef==probeRef) continue; //otherwise its the same object...
      float mass = (tagRef->p4()+probeRef->p4()).mag();
      if( ( mass>minMass_ || minMass_<0 ) && 
	  ( mass<maxMass_ || maxMass_<0 ) && 
	  ( !requireOpSign_ || tagRef->charge()!=probeRef->charge()) ){
	for(auto& filterHist : filterHists_){
	  filterHist.fillHists(*tagRef,*probeRef,event,setup,*trigEvtHandle);
	}	      
      }//end of t&p pair cuts
    }//end of probe loop
  }//end of tag loop
}

template <typename ObjType,typename ObjCollType> 
std::vector<edm::Ref<ObjCollType> > HLTDQMTagAndProbeEff<ObjType,ObjCollType>::
getPassingRefs(const edm::Handle<ObjCollType>& objCollHandle,
	       const trigger::TriggerEvent& trigEvt,
	       const std::vector<std::string>& filterNames,
	       const bool orFilters,
	       const edm::Handle<edm::ValueMap<bool> >& vidHandle,
	       const VarRangeCutColl<ObjType>& rangeCuts)
{
  std::vector<edm::Ref<ObjCollType> > passingRefs;
  for(size_t objNr=0;objNr<objCollHandle->size();objNr++){
    edm::Ref<ObjCollType> ref(objCollHandle,objNr);
    if(rangeCuts(*ref) && 
       hltdqm::passTrig(ref->eta(),ref->phi(),trigEvt,filterNames,orFilters,hltProcess_) && 
       (vidHandle.isValid()==false || (*vidHandle)[ref]==true)){
      passingRefs.push_back(ref);
    }
  }
  return passingRefs;
}
#endif
