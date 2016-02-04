#include "DQMOffline/Trigger/interface/EgHLTTrigTools.h"

#include "FWCore/ParameterSet/interface/Registry.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <boost/algorithm/string.hpp>
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
 

//this looks into the HLT config and fills a sorted vector with the last filter of all HLT triggers
//it assumes this filter is either the last (in the case of ES filters) or second to last in the sequence
void trigTools::getActiveFilters(const HLTConfigProvider& hltConfig,std::vector<std::string>& activeFilters)
{
  activeFilters.clear();

  for(size_t pathNr=0;pathNr<hltConfig.size();pathNr++){
    const std::string& pathName = hltConfig.triggerName(pathNr);
    if(pathName.find("HLT_")==0){ //hlt path as they all start with HLT_XXXX
  
      std::string lastFilter;
      const std::vector<std::string>& filters = hltConfig.moduleLabels(pathNr);
      if(!filters.empty()){
	if(filters.back()=="hltBoolEnd" && filters.size()>=2){
	  activeFilters.push_back(filters[filters.size()-2]); //2nd to last element is the last filter, useally the case as last is hltBool except for ES bits
	}else activeFilters.push_back(filters.back());
	//std::cout<<filters[filters.size()-2]<<std::endl;
      }
    }//end hlt path check
  }//end path loop over

  std::sort(activeFilters.begin(),activeFilters.end());
  
}
//----Morse test--------------
//want to grab filters based on name, hopefully be able to apply offline cuts
//this is first test to find photon30caloidl
//based on names as well
/*void trigTools::getPhoton30(const HLTConfigProvider& hltConfig,std::vector<std::string>& activeFilters)
  {
  
  activeFilters.clear();
  
  for(size_t pathNr=0;pathNr<hltConfig.size();pathNr++){
  const std::string& pathName = hltConfig.triggerName(pathNr);
  if(pathName.find("HLT_")==0){ //hlt path as they all start with HLT_XXXX
  if((pathName.find("Photon")==0 || pathName.find("Ele")==0 || pathName.find("EG")==0) && pathName.find("Photon30")==0 ){
  
  std::string lastFilter;
  const std::vector<std::string>& filters = hltConfig.moduleLabels(pathNr);
  if(!filters.empty()){
  if(filters.back()=="hltBoolEnd" && filters.size()>=2){
  activeFilters.push_back(filters[filters.size()-2]); //2nd to last element is the last filter, useally the case as last is hltBool except for ES bits
  //std::cout<<filters[filters.size()-2]<<std::endl;
  }else activeFilters.push_back(filters.back());
  }
  }
  }//end hlt path check
  }//end path loop over
  std::sort(activeFilters.begin(),activeFilters.end());
  }*/
//----------------------------

//this function will filter the inactive filternames
//it assumes the list of active filters is sorted   
//at some point this will be replaced with one line of fancy stl code but I want it to work now :)
void trigTools::filterInactiveTriggers(std::vector<std::string>& namesToFilter,const std::vector<std::string>& activeFilters)
{
  //tempory vector to store the filtered results
  std::vector<std::string> filteredNames;
  
  for(size_t inputFilterNr=0;inputFilterNr<namesToFilter.size();inputFilterNr++){
    if(std::binary_search(activeFilters.begin(),activeFilters.end(),namesToFilter[inputFilterNr])){
      filteredNames.push_back(namesToFilter[inputFilterNr]);
    }//std::cout<<filteredNames[inputFilterNr]<<std::endl;
  }
  
  namesToFilter.swap(filteredNames);
}

//input filters have format filter1:filter2, this checks both filters are active, rejects ones where both are not active
void trigTools::filterInactiveTightLooseTriggers(std::vector<std::string>& namesToFilter,const std::vector<std::string>& activeFilters)
{
  //tempory vector to store the filtered results
  std::vector<std::string> filteredNames;
  
  for(size_t inputFilterNr=0;inputFilterNr<namesToFilter.size();inputFilterNr++){
    std::vector<std::string> names;
    boost::split(names,namesToFilter[inputFilterNr],boost::is_any_of(std::string(":")));
    if(names.size()!=2) continue; //format incorrect, reject it
    if(std::binary_search(activeFilters.begin(),activeFilters.end(),names[0]) &&
       std::binary_search(activeFilters.begin(),activeFilters.end(),names[1])){ //both filters are valid
      filteredNames.push_back(namesToFilter[inputFilterNr]);
    }
  }
  
  namesToFilter.swap(filteredNames);
}

//a comparison functiod for std::pair<std::string,std::string> 
//this probably (infact must) exist elsewhere
class StringPairCompare {
public: 
  bool operator()(const std::pair<std::string,std::string>&lhs,
		  const std::pair<std::string,std::string>& rhs)const{return keyLess(lhs.first,rhs.first);}
  bool operator()(const std::pair<std::string,std::string>&lhs,
		  const std::pair<std::string,std::string>::first_type& rhs)const{return keyLess(lhs.first,rhs);}
  bool operator()(const std::pair<std::string,std::string>::first_type &lhs,
		  const std::pair<std::string,std::string>& rhs)const{return keyLess(lhs,rhs.first);}
private:
  bool keyLess(const std::pair<std::string,std::string>::first_type& k1,const std::pair<std::string,std::string>::first_type& k2)const{return k1<k2;}
};

void trigTools::translateFiltersToPathNames(const HLTConfigProvider& hltConfig,const std::vector<std::string>& filters,std::vector<std::string>& paths)
{
  
  paths.clear();
  std::vector<std::pair<std::string,std::string> > filtersAndPaths;

  for(size_t pathNr=0;pathNr<hltConfig.size();pathNr++){
    const std::string& pathName = hltConfig.triggerName(pathNr);
    if(pathName.find("HLT_")==0){ //hlt path as they all start with HLT_XXXX
  
      std::string lastFilter;
      const std::vector<std::string>& pathFilters = hltConfig.moduleLabels(pathNr);
      if(!pathFilters.empty()){
	if(pathFilters.back()=="hltBoolEnd" && pathFilters.size()>=2){
	  //2nd to last element is the last filter, useally the case as last is hltBool except for ES bits
	  filtersAndPaths.push_back(std::make_pair(pathFilters[pathFilters.size()-2],pathName));
	}else filtersAndPaths.push_back(std::make_pair(pathFilters.back(),pathName));
      }
    }//end hlt path check
  }//end path loop over

  std::sort(filtersAndPaths.begin(),filtersAndPaths.end(),StringPairCompare());
  
  for(size_t filterNr=0;filterNr<filters.size();filterNr++){
    typedef std::vector<std::pair<std::string,std::string> >::const_iterator VecIt;
    std::pair<VecIt,VecIt> searchResult = std::equal_range(filtersAndPaths.begin(),filtersAndPaths.end(),filters[filterNr],StringPairCompare());
    if(searchResult.first!=searchResult.second) paths.push_back(searchResult.first->second);
    else paths.push_back(filters[filterNr]);//if cant find the path, just  write the filter
    //---Morse-----
    //std::cout<<filtersAndPaths[filterNr].first<<"  "<<filtersAndPaths[filterNr].second<<std::endl;
    //-------------
  }

}

std::string trigTools::getL1SeedFilterOfPath(const HLTConfigProvider& hltConfig,const std::string& path)
{
  const std::vector<std::string>& modules = hltConfig.moduleLabels(path);

  for(size_t moduleNr=0;moduleNr<modules.size();moduleNr++){
    const std::string& moduleName=modules[moduleNr]; 
    if(moduleName.find("hltL1s")==0) return moduleName; //found l1 seed module
  }
  std::string dummy;
  return dummy;

}
  
//hunts for first instance of pattern EtX where X = a number of any length and returns X
float  trigTools::getEtThresFromName(const std::string& trigName)
{
  size_t etStrPos = trigName.find("Et");
  while(etStrPos!=std::string::npos && trigName.find_first_of("1234567890",etStrPos)!=etStrPos+2){
    etStrPos = trigName.find("Et",etStrPos+1);  
  }
  if(etStrPos!=std::string::npos && trigName.find_first_of("1234567890",etStrPos)==etStrPos+2){
    size_t endOfEtValStr = trigName.find_first_not_of("1234567890",etStrPos+2);  

    std::istringstream etValStr(trigName.substr(etStrPos+2,endOfEtValStr-etStrPos-2));
    float etVal;
    etValStr>> etVal;
    return etVal;
    
  }
  return 0;

}
