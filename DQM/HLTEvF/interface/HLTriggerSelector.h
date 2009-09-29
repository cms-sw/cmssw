#ifndef HLTriggerSelector_H
#define HLTriggerSelector_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "TPRegexp.h" 
#include "TString.h"

class HLTriggerSelector { 
 public:
  HLTriggerSelector(const edm::ParameterSet & iConfig){
    //if (iConfig.exists("HLTriggerSelector")){
    // const edm::ParameterSet & pset = iConfig.getParameter<edm::ParameterSet>("HLTriggerSelector");
      
    theTriggerResulTag = iConfig.getParameter< edm::InputTag >("TriggerResultsTag");
    std::vector<std::string> selectTriggers = iConfig.getParameter<std::vector<std::string> >("HLTPaths");
      
    //get the configuration
    HLTConfigProvider hltConfig;
    hltConfig.init(theTriggerResulTag.process());
    std::vector<std::string> validTriggerNames = hltConfig.triggerNames(); 
      
    bool goodToGo = false;
    //remove all path names that are not valid
    while(!goodToGo && selectTriggers.size()!=0){
      goodToGo=true;
      for (std::vector<std::string>::iterator j=selectTriggers.begin();j!=selectTriggers.end();++j){
	bool goodOne = false;
	//check if trigger name is valid
	//use of wildcard
	TPRegexp wildCard(*j);
	//std::cout << "wildCard.GetPattern() = " << wildCard.GetPattern() << std::endl;
	for (unsigned int i = 0; i != validTriggerNames.size(); ++i){
	  if (TString(validTriggerNames[i]).Contains(wildCard)){ 
	    goodOne = true;
	    if (find(theSelectTriggers.begin(),
		     theSelectTriggers.end(), 
		     validTriggerNames[i])==theSelectTriggers.end()){
	      //std::cout << "wildcard trigger = " << validTriggerNames[i] << std::endl;
	      theSelectTriggers.push_back( validTriggerNames[i] ); //add it after duplicate check.
	    }
	  }
	}
      }
    }//while
  }

  bool selectEvent(const edm::Event & iEvent){
    //auto accept if nothing was configured.
    if (theSelectTriggers.empty()) return true;

    edm::Handle<edm::TriggerResults> tResults;
    iEvent.getByLabel(theTriggerResulTag, tResults);
    if (tResults.failedToGet()){
      edm::LogError("HLTriggerSelector")
	<<"could not get the trigger results with tag: "<<theTriggerResulTag;
	return false;
    }
    
    theTriggerNames.init(*tResults);
    
    for (uint i=0;i!=theSelectTriggers.size();++i){
      uint index = theTriggerNames.triggerIndex( theSelectTriggers[i] );
      if ( index < tResults->size() && tResults->accept( index )) return true;
    }
    return false;
  }

  // private:
  edm::TriggerNames theTriggerNames ;
  edm::InputTag theTriggerResulTag;
  std::vector<std::string> theSelectTriggers;

};
#endif
