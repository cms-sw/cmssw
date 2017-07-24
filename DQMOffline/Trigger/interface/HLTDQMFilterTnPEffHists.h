#ifndef DQMOnline_Trigger_HLTDQMHistTnPColl_h
#define DQMOnline_Trigger_HLTDQMHistTnPColl_h

//********************************************************************************
//
// Description:
//   This class allows additional, collection specific requirements to be placed
//   on the tag electron. For example when measuring the second leg of 
//   an unseeded double electron trigger, you need to ensure the tag passes
//   the first leg of the trigger. 
// 
//   Some debate on whether this should inherit from or own a HLTDQMFilterEffHists
//   Because the fill interface is different, I eventually decided it owns a copy
//   rather than mucking around with inheritance
//
// Author : Sam Harper , RAL, May 2017
//
//***********************************************************************************

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMOffline/Trigger/interface/HLTDQMFilterEffHists.h"

#include <string>

template <typename TagType,typename ProbeType=TagType>
class HLTDQMFilterTnPEffHists  {
public:
  HLTDQMFilterTnPEffHists(const edm::ParameterSet& config,
			  const std::string& baseHistName,
			  const std::string& hltProcess):
    histColl_(config,baseHistName,hltProcess),
    tagExtraFilter_(config.getParameter<std::string>("tagExtraFilter")),
    hltProcess_(hltProcess){}

  static edm::ParameterSetDescription makePSetDescription(){
    edm::ParameterSetDescription desc = HLTDQMFilterEffHists<ProbeType>::makePSetDescription();
    desc.add<std::string>("tagExtraFilter","");
    return desc;
  }
  static edm::ParameterSetDescription makePSetDescriptionHistConfigs(){
    return HLTDQMFilterEffHists<ProbeType>::makePSetDescriptionHistConfigs();
  }

  void bookHists(DQMStore::IBooker& iBooker,const std::vector<edm::ParameterSet>& histConfigs){
    histColl_.bookHists(iBooker,histConfigs);
  }
  void fillHists(const TagType& tag,const ProbeType& probe,
		 const edm::Event& event,const edm::EventSetup& setup,
		 const trigger::TriggerEvent& trigEvt){
    if(tagExtraFilter_.empty() || hltdqm::passTrig(tag.eta(),tag.phi(),trigEvt,tagExtraFilter_,hltProcess_)){
      histColl_.fillHists(probe,event,setup,trigEvt);
    } 
  }
private:
  //these hists take the probe as input hence they are of ProbeType
  HLTDQMFilterEffHists<ProbeType> histColl_;
  std::string tagExtraFilter_;
  //really wondering whether to put an accessor to HLTDQMFilterEffHists for this
  //feels ineligant though so I made another copy here
  std::string hltProcess_; 
};



#endif
