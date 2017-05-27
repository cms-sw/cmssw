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



#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DQMOffline/Trigger/interface/UtilFuncs.h"
#include "DQMOffline/Trigger/interface/VarRangeCutColl.h"
#include "DQMOffline/Trigger/interface/HLTDQMFilterTnPEffHists.h"


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
  
  void bookHists(DQMStore::IBooker& iBooker);
  void fill(const edm::Event& event,const edm::EventSetup& setup);
  
private:
  std::vector<edm::Ref<ObjCollType> >
  getPassingRefs(const edm::Handle<ObjCollType>& objCollHandle,
		 const trigger::TriggerEvent& trigEvt,
		 const std::vector<std::string>& filterNames,
		 const edm::Handle<edm::ValueMap<bool> >& vidHandle,
		 const VarRangeCutColl<ObjType>& rangeCuts);
private:
  edm::EDGetTokenT<ObjCollType> objToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> trigEvtToken_;
  edm::EDGetTokenT<edm::TriggerResults> trigResultsToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> >tagVIDToken_;
  edm::EDGetTokenT<edm::ValueMap<bool> >probeVIDToken_;
  
  std::string hltProcess_;

  std::string tagTrigger_;
	    
  std::vector<std::string> tagFilters_;
  VarRangeCutColl<ObjType> tagRangeCuts_;

  std::vector<std::string> probeFilters_;
  VarRangeCutColl<ObjType> probeRangeCuts_;

  float minMass_;
  float maxMass_;
  bool requireOpSign_;

  std::vector<edm::ParameterSet> histConfigs_;
  std::vector<HLTDQMFilterTnPEffHists<ObjType> > filterHists_;
    
};

template <typename ObjType,typename ObjCollType> 
HLTDQMTagAndProbeEff<ObjType,ObjCollType>::HLTDQMTagAndProbeEff(const edm::ParameterSet& pset,edm::ConsumesCollector && cc):
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
  desc.add<edm::InputTag>("trigResults",edm::InputTag("TriggerResults","","HLT"));
  desc.add<edm::InputTag>("tagVIDCuts",edm::InputTag());
  desc.add<edm::InputTag>("probeVIDCuts",edm::InputTag());
  desc.add<std::string>("tagTrigger","");
  desc.add<std::vector<std::string> >("tagFilters",std::vector<std::string>());
  desc.add<std::vector<std::string> >("probeFilters",std::vector<std::string>());
  desc.add<double>("minMass");
  desc.add<double>("maxMass");
  desc.add<bool>("requireOpSign");
  desc.addVPSet("histConfigs",HLTDQMFilterTnPEffHists<ObjType>::makePSetDescriptionHistConfigs(),std::vector<edm::ParameterSet>()); 
  desc.addVPSet("filterConfigs",HLTDQMFilterTnPEffHists<ObjType>::makePSetDescription(),std::vector<edm::ParameterSet>()); 
  desc.add<std::string>("baseHistName");
  return desc;
}

template <typename ObjType,typename ObjCollType> 
std::vector<edm::Ref<ObjCollType> > HLTDQMTagAndProbeEff<ObjType,ObjCollType>::
getPassingRefs(const edm::Handle<ObjCollType>& objCollHandle,
	       const trigger::TriggerEvent& trigEvt,
	       const std::vector<std::string>& filterNames,
	       const edm::Handle<edm::ValueMap<bool> >& vidHandle,
	       const VarRangeCutColl<ObjType>& rangeCuts)
{
  std::vector<edm::Ref<ObjCollType> > passingRefs;
  for(size_t objNr=0;objNr<objCollHandle->size();objNr++){
    edm::Ref<ObjCollType> ref(objCollHandle,objNr);
    if(rangeCuts(*ref) && 
       hltdqm::passTrig(ref->eta(),ref->phi(),trigEvt,filterNames,hltProcess_) && 
       (vidHandle.isValid()==false || (*vidHandle)[ref]==true)){
      passingRefs.push_back(ref);
    }
  }
  return passingRefs;
}


template <typename ObjType,typename ObjCollType> 
void HLTDQMTagAndProbeEff<ObjType,ObjCollType>::fill(const edm::Event& event,const edm::EventSetup& setup)
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
  if(hltdqm::passTrig(tagTrigger_,trigNames,*trigResultsHandle)==false) return;
  
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
	for(auto& filterHist : filterHists_){
	  filterHist.fillHists(*tagRef,*probeRef,event,setup,*trigEvtHandle);
	}	      
      }//end of t&p pair cuts
    }//end of probe loop
  }//end of tag loop
}

template <typename ObjType,typename ObjCollType> 
void HLTDQMTagAndProbeEff<ObjType,ObjCollType>::bookHists(DQMStore::IBooker& iBooker)
{
  for(auto& filterHist:  filterHists_) filterHist.bookHists(iBooker,histConfigs_);
}

#endif
