#ifndef DQMOFFLINE_TRIGGER_EGHLTTRIGTOOLS
#define DQMOFFLINE_TRIGGER_EGHLTTRIGTOOLS

#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"

class HLTConfigProvider;

namespace egHLT {
  
  namespace trigTools {
    TrigCodes::TrigBitSet getFiltersPassed(const std::vector<std::pair<std::string,int> >& filters,const trigger::TriggerEvent* trigEvt,const std::string& hltTag,const TrigCodes& trigCodes);
    
    template<class T> void setFiltersObjPasses(std::vector<T>& objs,const std::vector<std::string>& filters,const std::vector<std::pair<std::string,std::string> >& l1PreAndSeedFilters,const TrigCodes::TrigBitSet& evtTrigBits, const TrigCodes& trigCodes, const trigger::TriggerEvent* trigEvt,const std::string& hltTag );
    
    template<class T, class U> void fillHLTposition(T& obj,U& hltData,const std::vector<std::string>& filters,const trigger::TriggerEvent* trigEvt,const std::string& hltTag );
    int getMinNrObjsRequiredByFilter(const std::string& filterName); //slow function, call at begin job and cache results

    //reads hlt config and works out which are the active last filters stored in trigger summary, is sorted
    void getActiveFilters(const HLTConfigProvider& hltConfig,std::vector<std::string>& activeFilters,std::vector<std::string>& activeEleFilters,std::vector<std::string>& activeEle2LegFilters,std::vector<std::string>& activePhoFilters,std::vector<std::string>& activePho2LegFilters);
    //---Morse test--------
    //void getPhoton30(const HLTConfigProvider& hltConfig,std::vector<std::string>& activeFilters);
    //------------------
    //filters a list of filternames removing any filters which are not in active filters, assumes active filters is sorted
    void filterInactiveTriggers(std::vector<std::string>& namesToFilter,std::vector<std::string>& activeFilters);
    //filters a list of filterName1:filterName2 removing any entry for which either filter is not in activeFilters, assumes active filters is sorted
    void filterInactiveTightLooseTriggers(std::vector<std::string>& namesToFilter,const std::vector<std::string>& activeFilters);

    void translateFiltersToPathNames(const HLTConfigProvider& hltConfig,const std::vector<std::string>& filters,std::vector<std::string>& paths);
    std::string getL1SeedFilterOfPath(const HLTConfigProvider& hltConfig,const std::string& path);

    //looks for string Et and then looks for a number after that (currently the standard of all E/g triggers)
    //returns 0 if unsuccessful
    float getEtThresFromName(const std::string& trigName);
    float getSecondEtThresFromName(const std::string& trigName);
  }
  
  //I have the horrible feeling that I'm converting into an intermediatry format and then coverting back again
  //Okay how this works
  //1) create a TrigBitSet for each particle set to 0 initally
  //2) loop over each filter, for each particle that passes the filter, set the appropriate bit in the TrigBitSet
  //3) after that, loop over each particle setting that its TrigBitSet which has been calculated
  //4) because L1 pre-scaled paths are special, we only set those if an event wide trigger has been set
  template <class T>
  void trigTools::setFiltersObjPasses(std::vector<T>& particles,const std::vector<std::string>& filters,
				      const std::vector<std::pair<std::string,std::string> >& l1PreAndSeedFilters,
				      const TrigCodes::TrigBitSet& evtTrigBits,
              const TrigCodes& trigCodes,
				      const trigger::TriggerEvent* trigEvt,const std::string& hltTag)
  {
    std::vector<TrigCodes::TrigBitSet> partTrigBits(particles.size());
    const double maxDeltaR=0.1;
    for(size_t filterNrInVec=0;filterNrInVec<filters.size();filterNrInVec++){
      size_t filterNrInEvt = trigEvt->filterIndex(edm::InputTag(filters[filterNrInVec],"",hltTag).encode());
      const TrigCodes::TrigBitSet filterCode = trigCodes.getCode(filters[filterNrInVec].c_str());
      
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
    
    //okay the first element is the key, the second is the filter that exists in trigger event
    for(size_t l1FilterNrInVec=0;l1FilterNrInVec<l1PreAndSeedFilters.size();l1FilterNrInVec++){
      const TrigCodes::TrigBitSet filterCode = trigCodes.getCode(l1PreAndSeedFilters[l1FilterNrInVec].first.c_str());
      if((filterCode&evtTrigBits)==filterCode){ //check that filter has fired in the event
   
	size_t filterNrInEvt = trigEvt->filterIndex(edm::InputTag(l1PreAndSeedFilters[l1FilterNrInVec].second,"",hltTag).encode());
	
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
      }//end check if path has fired in the event
    }//end loop over all filters 
    
    for(size_t partNr=0;partNr<particles.size();partNr++) particles[partNr].setTrigBits(partTrigBits[partNr]);
    
  }


 template <class T, class U>
  void trigTools::fillHLTposition(T& particle,
				  U& hltData,
				  const std::vector<std::string>& filters,
				  const trigger::TriggerEvent* trigEvt,
				  const std::string& hltTag)
{
  std::vector<TrigCodes::TrigBitSet> partTrigBits(1);
  const double maxDeltaR=0.1;
  for(size_t filterNrInVec=0;filterNrInVec<filters.size();filterNrInVec++){
    size_t filterNrInEvt = trigEvt->filterIndex(edm::InputTag(filters[filterNrInVec],"",hltTag).encode());
    //const TrigCodes::TrigBitSet filterCode = trigCodes.getCode(filters[filterNrInVec].c_str()); 
    if(filterNrInEvt<trigEvt->sizeFilters()){ //filter found in event, something passes it
      const trigger::Keys& trigKeys = trigEvt->filterKeys(filterNrInEvt);  //trigger::Keys is actually a vector<uint16_t> holding the position of trigger objects in the trigger collection passing the filter
      const trigger::TriggerObjectCollection & trigObjColl(trigEvt->getObjects());
      for(trigger::Keys::const_iterator keyIt=trigKeys.begin();keyIt!=trigKeys.end();++keyIt){
	float trigObjEta = trigObjColl[*keyIt].eta();
	float trigObjPhi = trigObjColl[*keyIt].phi();
	float trigObjE = trigObjColl[*keyIt].energy();
	if (reco::deltaR(particle.superCluster()->eta(),particle.superCluster()->phi(),trigObjEta,trigObjPhi) < maxDeltaR){
	  hltData.HLTeta=trigObjEta;
	  hltData.HLTphi=trigObjPhi;
	  hltData.HLTenergy=trigObjE;
	}//end dR<maxDeltaR trig obj match test
      }//end loop over all objects passing filter`
    }//end check if filter is present
  }//end check if path has fired in the event
}//end loop over all filters 
  
}//end namespace declaration

#endif
  
