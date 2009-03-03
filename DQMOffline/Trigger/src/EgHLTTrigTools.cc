#include "DQMOffline/Trigger/interface/EgHLTTrigTools.h"


using namespace egHLT;

TrigCodes::TrigBitSet trigTools::getFiltersPassed(const std::vector<std::string>& filters,const trigger::TriggerEvent* trigEvt,const std::string& hltTag)
{
  TrigCodes::TrigBitSet evtTrigs;
  for(size_t filterNrInVec=0;filterNrInVec<filters.size();filterNrInVec++){
    size_t filterNrInEvt = trigEvt->filterIndex(edm::InputTag(filters[filterNrInVec],"",hltTag).encode());
    const TrigCodes::TrigBitSet filterCode = TrigCodes::getCode(filters[filterNrInVec].c_str());
    if(filterNrInEvt<trigEvt->sizeFilters()){ //filter found in event, something passes it
      evtTrigs |=filterCode; //if something passes it add to the event trigger bits
    }//end check if filter is present
  }//end loop over all filters

  return evtTrigs;

}
