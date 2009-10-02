#include "DQMOffline/Trigger/interface/EgHLTTrigTools.h"

#include "FWCore/ParameterSet/interface/Registry.h"


using namespace egHLT;

TrigCodes::TrigBitSet trigTools::getFiltersPassed(const std::vector<std::pair<std::string,int> >& filters,const trigger::TriggerEvent* trigEvt,const std::string& hltTag)
{
  TrigCodes::TrigBitSet evtTrigs;
  for(size_t filterNrInVec=0;filterNrInVec<filters.size();filterNrInVec++){
    size_t filterNrInEvt = trigEvt->filterIndex(edm::InputTag(filters[filterNrInVec].first,"",hltTag).encode());
    const TrigCodes::TrigBitSet filterCode = TrigCodes::getCode(filters[filterNrInVec].first.c_str());
    if(filterNrInEvt<trigEvt->sizeFilters()){ //filter found in event, however this only means that something passed the previous filter
      const trigger::Keys& trigKeys = trigEvt->filterKeys(filterNrInEvt);
      if(static_cast<int>(trigKeys.size())>=filters[filterNrInVec].second){
	evtTrigs |=filterCode; //filter was passed
      }
    }//end check if filter is present
  }//end loop over all filters

  return evtTrigs;

}


//this function runs over all parameter sets for every module that has ever run on an event in this job
//it looks for the specified filter module
//and returns the minimum number of objects required to pass the filter, -1 if its not found
//which is either the ncandcut or MinN parameter in the filter config
//assumption: nobody will ever change MinN or ncandcut without changing the filter name
//as this just picks the first module name and if 2 different versions of HLT were run with the filter having
//a different min obj required in the two versions, this may give the wrong answer
int trigTools::getMinNrObjsRequiredByFilter(const std::string& filterName)
{
 
  //will return out of for loop once its found it to save time
  const edm::pset::Registry* psetRegistry = edm::pset::Registry::instance();
  if(psetRegistry==NULL) return -1;
  for(edm::pset::Registry::const_iterator psetIt=psetRegistry->begin();psetIt!=psetRegistry->end();++psetIt){ //loop over every pset for every module ever run
    const std::map<std::string,edm::Entry>& mapOfPara  = psetIt->second.tbl(); //contains the parameter name and value for all the parameters of the pset
    const std::map<std::string,edm::Entry>::const_iterator itToModLabel = mapOfPara.find("@module_label"); 
    if(itToModLabel!=mapOfPara.end()){
      if(itToModLabel->second.getString()==filterName){ //moduleName is the filter name, we have found filter, we will now return something
	std::map<std::string,edm::Entry>::const_iterator itToCandCut = mapOfPara.find("ncandcut");
	if(itToCandCut!=mapOfPara.end() && itToCandCut->second.typeCode()=='I') return itToCandCut->second.getInt32();
	else{ //checks if MinN exists and is int32, if not return -1
	  itToCandCut = mapOfPara.find("MinN");
	  if(itToCandCut!=mapOfPara.end() && itToCandCut->second.typeCode()=='I') return itToCandCut->second.getInt32();
	  else return -1;
	}
      }
      
    }
  }
  return -1;
}
