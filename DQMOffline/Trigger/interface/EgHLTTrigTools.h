#ifndef DQMOFFLINE_TRIGGER_EGHLTTRIGTOOLS
#define DQMOFFLINE_TRIGGER_EGHLTTRIGTOOLS

#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace egHLT {
  
  namespace trigTools {
    TrigCodes::TrigBitSet getFiltersPassed(const std::vector<std::pair<std::string,int> >& filters,const trigger::TriggerEvent* trigEvt,const std::string& hltTag);
    template<class T> void setFiltersObjPasses(std::vector<T>& objs,const std::vector<std::string>& filters,const trigger::TriggerEvent* trigEvt,const std::string& hltTag );
    int getMinNrObjsRequiredByFilter(const std::string& filterName); //slow function, call at begin job and cache results
  }
  
  //I have the horrible feeling that I'm converting into an intermediatry format and then coverting back again
  //Okay how this works
  //1) create a TrigBitSet for each particle set to 0 initally
  //2) loop over each filter, for each particle that passes the filter, set the appropriate bit in the TrigBitSet
  //3) after that, loop over each particle setting the its TrigBitSet which has been calculated
  template <class T>
  void trigTools::setFiltersObjPasses(std::vector<T>& particles,const std::vector<std::string>& filters,const trigger::TriggerEvent* trigEvt,const std::string& hltTag)
  {
    std::vector<TrigCodes::TrigBitSet> partTrigBits(particles.size());
    const double maxDeltaR=0.1;
    for(size_t filterNrInVec=0;filterNrInVec<filters.size();filterNrInVec++){
      size_t filterNrInEvt = trigEvt->filterIndex(edm::InputTag(filters[filterNrInVec],"",hltTag).encode());
      const TrigCodes::TrigBitSet filterCode = TrigCodes::getCode(filters[filterNrInVec].c_str());
      
      if(filterNrInEvt<trigEvt->sizeFilters()){ //filter found in event, something passes it
	const trigger::Keys& trigKeys = trigEvt->filterKeys(filterNrInEvt);  //trigger::Keys is actually a vector<uint16_t> holding the position of trigger objects in the trigger collection passing the filter
	const trigger::TriggerObjectCollection & trigObjColl(trigEvt->getObjects());
	for(size_t partNr=0;partNr<particles.size();partNr++){
	  for(trigger::Keys::const_iterator keyIt=trigKeys.begin();keyIt!=trigKeys.end();++keyIt){
	    float trigObjEta = trigObjColl[*keyIt].eta();
	    float trigObjPhi = trigObjColl[*keyIt].phi();
	    if (reco::deltaR(particles[partNr].eta(),particles[partNr].phi(),trigObjEta,trigObjPhi) < maxDeltaR){
	    partTrigBits[partNr] |= filterCode;
	    }//end dR<maxDeltaR trig obj match test
	  }//end loop over all objects passing filter
	}//end loop over particles
      }//end check if filter is present
    }//end loop over all filters
    
    for(size_t partNr=0;partNr<particles.size();partNr++) particles[partNr].setTrigBits(partTrigBits[partNr]);
    
  }
}  
#endif
  
