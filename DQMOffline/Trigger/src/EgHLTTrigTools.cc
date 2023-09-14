#include "DQMOffline/Trigger/interface/EgHLTTrigTools.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <boost/algorithm/string.hpp>
using namespace egHLT;

TrigCodes::TrigBitSet trigTools::getFiltersPassed(const std::vector<std::pair<std::string, int> >& filters,
                                                  const trigger::TriggerEvent* trigEvt,
                                                  const std::string& hltTag,
                                                  const TrigCodes& trigCodes) {
  TrigCodes::TrigBitSet evtTrigs;
  for (auto const& filter : filters) {
    size_t filterNrInEvt = trigEvt->filterIndex(edm::InputTag(filter.first, "", hltTag));
    const TrigCodes::TrigBitSet filterCode = trigCodes.getCode(filter.first.c_str());
    if (filterNrInEvt <
        trigEvt
            ->sizeFilters()) {  //filter found in event, however this only means that something passed the previous filter
      const trigger::Keys& trigKeys = trigEvt->filterKeys(filterNrInEvt);
      if (static_cast<int>(trigKeys.size()) >= filter.second) {
        evtTrigs |= filterCode;  //filter was passed
      }
    }  //end check if filter is present
  }    //end loop over all filters

  return evtTrigs;
}

//this function runs over all parameter sets for every module that has ever run on an event in this job
//it looks for the specified filter module
//and returns the minimum number of objects required to pass the filter, -1 if its not found
//which is either the ncandcut or MinN parameter in the filter config
//assumption: nobody will ever change MinN or ncandcut without changing the filter name
//as this just picks the first module name and if 2 different versions of HLT were run with the filter having
//a different min obj required in the two versions, this may give the wrong answer
std::vector<int> trigTools::getMinNrObjsRequiredByFilter(const std::vector<std::string>& filterNames) {
  std::vector<int> retVal(filterNames.size(), -2);
  const std::string mag0("@module_label");
  const std::string mag1("ncandcut");
  const std::string mag2("nZcandcut");
  const std::string mag3("MinN");
  const std::string mag4("minN");

  std::vector<std::string> filterEntryStrings;
  filterEntryStrings.reserve(filterNames.size());
  for (auto const& filterName : filterNames) {
    const edm::Entry filterEntry(mag0, filterName, true);
    filterEntryStrings.push_back(filterEntry.toString());
  }

  //will return out of for loop once its found it to save time
  const edm::pset::Registry* psetRegistry = edm::pset::Registry::instance();
  if (psetRegistry == nullptr) {
    retVal = std::vector<int>(filterNames.size(), -1);
    return retVal;
  }
  for (auto& psetIt : *psetRegistry) {  //loop over every pset for every module ever run
    const auto& mapOfPara =
        psetIt.second.tbl();  //contains the parameter name and value for all the parameters of the pset
    const auto itToModLabel = mapOfPara.find(mag0);
    if (itToModLabel != mapOfPara.end()) {
      std::string itString = itToModLabel->second.toString();

      for (unsigned int i = 0; i < filterNames.size(); i++) {
        if (retVal[i] == -1)
          continue;  //already done

        if (itString ==
            filterEntryStrings[i]) {  //moduleName is the filter name, we have found filter, we will now return something
          auto itToCandCut = mapOfPara.find(mag1);
          if (itToCandCut != mapOfPara.end() && itToCandCut->second.typeCode() == 'I')
            retVal[i] = itToCandCut->second.getInt32();
          else {  //checks if nZcandcut exists and is int32, if not return -1
            itToCandCut = mapOfPara.find(mag2);
            if (itToCandCut != mapOfPara.end() && itToCandCut->second.typeCode() == 'I')
              retVal[i] = itToCandCut->second.getInt32();
            else {  //checks if MinN exists and is int32, if not return -1
              itToCandCut = mapOfPara.find(mag3);
              if (itToCandCut != mapOfPara.end() && itToCandCut->second.typeCode() == 'I')
                retVal[i] = itToCandCut->second.getInt32();
              else {  //checks if minN exists and is int32, if not return -1
                itToCandCut = mapOfPara.find(mag4);
                if (itToCandCut != mapOfPara.end() && itToCandCut->second.typeCode() == 'I')
                  retVal[i] = itToCandCut->second.getInt32();
                else
                  retVal[i] = -1;
              }
            }
          }
        }
      }
    }
  }
  for (unsigned int i = 0; i < filterNames.size(); i++)
    if (retVal[i] == -2)
      retVal[i] = -1;
  return retVal;
}

//this looks into the HLT config and fills a sorted vector with the last filter of all HLT triggers
//it assumes this filter is either the last (in the case of ES filters) or second to last in the sequence
/*void trigTools::getActiveFilters(const HLTConfigProvider& hltConfig,std::vector<std::string>& activeFilters)
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
  
  }*/
//----Morse--------------
//want to grab all modules with saveTags==true for e/g paths
//veto x-triggers, which will be handled by PAGs
//first step towards automation; for now it is just used to check against filtersToMon
//should have some overhead but they will be filtered out by filterInactiveTriggers anyway
void trigTools::getActiveFilters(const HLTConfigProvider& hltConfig,
                                 std::vector<std::string>& activeFilters,
                                 std::vector<std::string>& activeEleFilters,
                                 std::vector<std::string>& activeEle2LegFilters,
                                 std::vector<std::string>& activePhoFilters,
                                 std::vector<std::string>& activePho2LegFilters) {
  activeFilters.clear();
  activeEleFilters.clear();
  activeEle2LegFilters.clear();
  activePhoFilters.clear();
  activePho2LegFilters.clear();

  for (size_t pathNr = 0; pathNr < hltConfig.size(); pathNr++) {
    const std::string& pathName = hltConfig.triggerName(pathNr);

    if (pathName.find("HLT_") == 0) {  //hlt path as they all start with HLT_XXXX
      if ((pathName.find("Photon") == 4 || pathName.find("Ele") == 4 || pathName.find("EG") != pathName.npos ||
           pathName.find("PAPhoton") == 4 || pathName.find("PAEle") == 4 || pathName.find("PASinglePhoton") == 4 ||
           pathName.find("HIPhoton") == 4 || pathName.find("HIEle") == 4 || pathName.find("HISinglePhoton") == 4 ||
           pathName.find("Activity") == 4 || pathName.find("Physics") == 4 || pathName.find("DiSC") == 4) &&
          (pathName.find("Jet") == pathName.npos && pathName.find("Muon") == pathName.npos &&
           pathName.find("Tau") == pathName.npos && pathName.find("HT") == pathName.npos &&
           pathName.find("MR") == pathName.npos && pathName.find("LEITI") == pathName.npos &&
           pathName.find("Jpsi") == pathName.npos && pathName.find("Ups") == pathName.npos)) {  //veto x-triggers
        //std::string lastFilter;
        const std::vector<std::string>& filters = hltConfig.saveTagsModules(pathNr);

        //std::cout<<"Number of prescale sets: "<<hltConfig.prescaleSize()<<std::endl;
        //std::cout<<std::endl<<"Path Name: "<<pathName<<"   Prescale: "<<hltConfig.prescaleValue(1,pathName)<<std::endl;

        if (!filters.empty()) {
          //std::cout<<"Path Name: "<<pathName<<std::endl;
          //if(filters.back()=="hltBoolEnd" && filters.size()>=2){
          std::vector<int> minNRFFCache = getMinNrObjsRequiredByFilter(filters);

          for (size_t filter = 0; filter < filters.size(); filter++) {
            //std::cout << filters[filter] << std::endl;
            if (filters[filter].find("Filter") !=
                filters[filter].npos) {  //keep only modules that contain the word "Filter"
              //std::cout<<"  Module Name: "<<filters[filter]<<" filter#: "<<int(filter)<<"/"<<filters.size()<<" ncandcut: "<<trigTools::getMinNrObjsRequiredByFilter(filters[filter])<<std::endl;
              int minNRFF = minNRFFCache[filter];
              int minNRFFP1 = -99;
              if (filter < filters.size() - 1)
                minNRFFP1 = minNRFFCache[filter + 1];
              if (  //keep only the last filter and the last one with ncandcut==1 (for di-object triggers)
                  (minNRFF == 1 && minNRFFP1 == 2) ||
                  (minNRFF == 1 && minNRFFP1 == 1 && filters[filter + 1].find("Mass") != filters[filter + 1].npos) ||
                  (minNRFF == 1 && minNRFFP1 == 1 && filters[filter + 1].find("FEM") != filters[filter + 1].npos) ||
                  (minNRFF == 1 && minNRFFP1 == 1 && filters[filter + 1].find("PFMT") != filters[filter + 1].npos) ||
                  filter == filters.size() - 1) {
                activeFilters.push_back(filters[filter]);  //saves all modules with saveTags=true

                //std::cout<<"  Module Name: "<<filters[filter]<<" filter#: "<<int(filter)<<"/"<<filters.size()<<" ncandcut: "<<trigTools::getMinNrObjsRequiredByFilter(filters[filter])<<std::endl;
                if (pathName.find("Photon") != pathName.npos || pathName.find("Activity") != pathName.npos ||
                    pathName.find("Physics") != pathName.npos || pathName.find("DiSC") == 4) {
                  activePhoFilters.push_back(filters[filter]);  //saves all "Photon" paths into photon set
                  int posPho = pathName.find("Pho") + 1;
                  if (pathName.find("Pho", posPho) != pathName.npos || pathName.find("SC", posPho) != pathName.npos) {
                    //This saves all "x_Photon_x_Photon_x" and "x_Photon_x_SC_x" path filters into 2leg photon set
                    activePho2LegFilters.push_back(filters[filter]);
                    //std::cout<<"Pho2LegPath: "<<pathName<<std::endl;
                  }
                }
                if (pathName.find("Ele") != pathName.npos || pathName.find("Activity") != pathName.npos ||
                    pathName.find("Physics") != pathName.npos) {
                  activeEleFilters.push_back(filters[filter]);  //saves all "Ele" paths into electron set
                  int posEle = pathName.find("Ele") + 1;
                  if (pathName.find("Ele", posEle) != pathName.npos || pathName.find("SC", posEle) != pathName.npos) {
                    if ((minNRFF == 1 && minNRFFP1 == 2) ||
                        (minNRFF == 1 && minNRFFP1 == 1 &&
                         filters[filter + 1].find("Mass") != filters[filter + 1].npos) ||
                        (minNRFF == 1 && minNRFFP1 == 1 &&
                         filters[filter + 1].find("SC") != filters[filter + 1].npos) ||
                        (minNRFF == 1 && minNRFFP1 == 1 &&
                         filters[filter + 1].find("FEM") != filters[filter + 1].npos)) {
                      //This saves all "x_Ele_x_Ele_x" and "x_Ele_x_SC_x" path filters into 2leg electron set
                      activeEle2LegFilters.push_back(filters[filter] + "::" + filters[filter + 1]);
                      //std::cout<<"Ele2LegPath: "<<pathName<<std::endl;
                    }
                  }
                }
              }
            }
          }
          //std::cout<<filters[filters.size()-2]<<std::endl;
          //}else activeFilters.push_back(filters.back());
        }
      }
    }  //end hlt path check
  }    //end path loop over
  /*for(size_t i=0;i<activeEle2LegFilters.size();i++){
    std::cout<<"Leg1: "<<activeEle2LegFilters[i].substr(0,activeEle2LegFilters[i].find("::"))<<std::endl;
    std::cout<<"Leg2: "<<activeEle2LegFilters[i].substr(activeEle2LegFilters[i].find("::")+2)<<std::endl<<std::endl;
    }*/
  std::sort(activeFilters.begin(), activeFilters.end());
  std::sort(activeEleFilters.begin(), activeEleFilters.end());
  std::sort(activeEle2LegFilters.begin(), activeEle2LegFilters.end());
  std::sort(activePhoFilters.begin(), activePhoFilters.end());
}
//----------------------------

//this function will filter the inactive filternames
//it assumes the list of active filters is sorted
//at some point this will be replaced with one line of fancy stl code but I want it to work now :)
void trigTools::filterInactiveTriggers(std::vector<std::string>& namesToFilter,
                                       std::vector<std::string>& activeFilters) {
  //tempory vector to store the filtered results
  std::vector<std::string> filteredNames;
  /*
  for(size_t inputFilterNr=0;inputFilterNr<namesToFilter.size();inputFilterNr++){
    if(std::binary_search(activeFilters.begin(),activeFilters.end(),namesToFilter[inputFilterNr])){
      filteredNames.push_back(namesToFilter[inputFilterNr]);
    }//std::cout<<filteredNames[inputFilterNr]<<std::endl;
  }
  */
  //namesToFilter.swap(activeFilters);
  filteredNames = activeFilters;
  namesToFilter.swap(filteredNames);
}

//input filters have format filter1:filter2, this checks both filters are active, rejects ones where both are not active
void trigTools::filterInactiveTightLooseTriggers(std::vector<std::string>& namesToFilter,
                                                 const std::vector<std::string>& activeFilters) {
  //tempory vector to store the filtered results
  std::vector<std::string> filteredNames;

  for (auto& inputFilterNr : namesToFilter) {
    std::vector<std::string> names;
    boost::split(names, inputFilterNr, boost::is_any_of(std::string(":")));
    if (names.size() != 2)
      continue;  //format incorrect, reject it
    if (std::binary_search(activeFilters.begin(), activeFilters.end(), names[0]) &&
        std::binary_search(activeFilters.begin(), activeFilters.end(), names[1])) {  //both filters are valid
      filteredNames.push_back(inputFilterNr);
    }
  }

  namesToFilter.swap(filteredNames);
}

//a comparison functiod for std::pair<std::string,std::string>
//this probably (infact must) exist elsewhere
class StringPairCompare {
public:
  bool operator()(const std::pair<std::string, std::string>& lhs,
                  const std::pair<std::string, std::string>& rhs) const {
    return keyLess(lhs.first, rhs.first);
  }
  bool operator()(const std::pair<std::string, std::string>& lhs,
                  const std::pair<std::string, std::string>::first_type& rhs) const {
    return keyLess(lhs.first, rhs);
  }
  bool operator()(const std::pair<std::string, std::string>::first_type& lhs,
                  const std::pair<std::string, std::string>& rhs) const {
    return keyLess(lhs, rhs.first);
  }

private:
  bool keyLess(const std::pair<std::string, std::string>::first_type& k1,
               const std::pair<std::string, std::string>::first_type& k2) const {
    return k1 < k2;
  }
};

void trigTools::translateFiltersToPathNames(const HLTConfigProvider& hltConfig,
                                            const std::vector<std::string>& filters,
                                            std::vector<std::string>& paths) {
  paths.clear();
  std::vector<std::pair<std::string, std::string> > filtersAndPaths;

  for (size_t pathNr = 0; pathNr < hltConfig.size(); pathNr++) {
    const std::string& pathName = hltConfig.triggerName(pathNr);
    if (pathName.find("HLT_") == 0) {  //hlt path as they all start with HLT_XXXX

      std::string lastFilter;
      const std::vector<std::string>& pathFilters = hltConfig.moduleLabels(pathNr);
      if (!pathFilters.empty()) {
        if (pathFilters.back() == "hltBoolEnd" && pathFilters.size() >= 2) {
          //2nd to last element is the last filter, useally the case as last is hltBool except for ES bits
          filtersAndPaths.push_back(std::make_pair(pathFilters[pathFilters.size() - 2], pathName));
        } else
          filtersAndPaths.push_back(std::make_pair(pathFilters.back(), pathName));
      }
    }  //end hlt path check
  }    //end path loop over

  std::sort(filtersAndPaths.begin(), filtersAndPaths.end(), StringPairCompare());

  for (auto const& filter : filters) {
    typedef std::vector<std::pair<std::string, std::string> >::const_iterator VecIt;
    std::pair<VecIt, VecIt> searchResult =
        std::equal_range(filtersAndPaths.begin(), filtersAndPaths.end(), filter, StringPairCompare());
    if (searchResult.first != searchResult.second)
      paths.push_back(searchResult.first->second);
    else
      paths.push_back(filter);  //if cant find the path, just  write the filter
    //---Morse-----
    //std::cout<<filtersAndPaths[filterNr].first<<"  "<<filtersAndPaths[filterNr].second<<std::endl;
    //-------------
  }
}

std::string trigTools::getL1SeedFilterOfPath(const HLTConfigProvider& hltConfig, const std::string& path) {
  const std::vector<std::string>& modules = hltConfig.moduleLabels(path);

  for (auto const& moduleName : modules) {
    if (moduleName.find("hltL1s") == 0)
      return moduleName;  //found l1 seed module
  }
  std::string dummy;
  return dummy;
}

//hunts for first instance of pattern EtX where X = a number of any length and returns X
float trigTools::getEtThresFromName(const std::string& trigName) {
  size_t etStrPos = trigName.find("Et");
  while (etStrPos != std::string::npos && trigName.find_first_of("1234567890", etStrPos) != etStrPos + 2) {
    etStrPos = trigName.find("Et", etStrPos + 1);
  }
  if (etStrPos != std::string::npos && trigName.find_first_of("1234567890", etStrPos) == etStrPos + 2) {
    size_t endOfEtValStr = trigName.find_first_not_of("1234567890", etStrPos + 2);

    std::istringstream etValStr(trigName.substr(etStrPos + 2, endOfEtValStr - etStrPos - 2));
    float etVal;
    etValStr >> etVal;
    return etVal;
  }
  return 0;
}

//hunts for second instance of pattern X where X = a number of any length and returns X
//This has gotten ridiculously more complicated now that filters do not have the "Et" string in them
float trigTools::getSecondEtThresFromName(
    const std::string& trigName) {  //std::cout<<"What the heck is this trigName?:"<<trigName<<std::endl;
  bool isEle = false, isPhoton = false, isEG = false, isEle2 = false, isPhoton2 = false, isEG2 = false, isSC2 = false;
  size_t etStrPos = trigName.npos;
  if (trigName.find("Ele") < trigName.find("Photon") && trigName.find("Ele") < trigName.find("EG")) {
    etStrPos = trigName.find("Ele");
    isEle = true;
  } else if (trigName.find("EG") < trigName.find("Photon") && trigName.find("EG") < trigName.find("Ele")) {
    etStrPos = trigName.find("EG");
    isEG = true;
  } else if (trigName.find("Photon") < trigName.find("Ele") && trigName.find("Photon") < trigName.find("EG")) {
    etStrPos = trigName.find("Photon");
    isPhoton = true;
  }
  //size_t etStrPos = trigName.find("Et");
  //std::cout<<"Got Original Et spot; etStrPos="<<etStrPos<<std::endl;
  /*while(etStrPos!=std::string::npos && trigName.find_first_of("1234567890",etStrPos)!=etStrPos+2){
    etStrPos = trigName.find("Et",etStrPos+1);//std::cout<<"Got first Et spot; etStrPos="<<etStrPos<<std::endl;
    }*/
  if (etStrPos != trigName.npos &&
      (trigName.find("Ele", etStrPos + 1) != trigName.npos || trigName.find("EG", etStrPos + 1) != trigName.npos ||
       trigName.find("Photon", etStrPos + 1) != trigName.npos || trigName.find("SC", etStrPos + 1) != trigName.npos)) {
    if (trigName.find("Ele", etStrPos + 1) < trigName.find("Photon", etStrPos + 1) &&
        trigName.find("Ele", etStrPos + 1) < trigName.find("EG", etStrPos + 1) &&
        trigName.find("Ele", etStrPos + 1) < trigName.find("SC", etStrPos + 1)) {
      etStrPos = trigName.find("Ele", etStrPos + 1);
      isEle2 = true;
    } else if (trigName.find("EG", etStrPos + 1) < trigName.find("Photon", etStrPos + 1) &&
               trigName.find("EG", etStrPos + 1) < trigName.find("Ele", etStrPos + 1) &&
               trigName.find("EG", etStrPos + 1) < trigName.find("SC", etStrPos + 1)) {
      etStrPos = trigName.find("EG", etStrPos + 1);
      isEG2 = true;
    } else if (trigName.find("Photon", etStrPos + 1) < trigName.find("EG", etStrPos + 1) &&
               trigName.find("Photon", etStrPos + 1) < trigName.find("Ele", etStrPos + 1) &&
               trigName.find("Photon", etStrPos + 1) < trigName.find("SC", etStrPos + 1)) {
      etStrPos = trigName.find("Photon", etStrPos + 1);
      isPhoton2 = true;
    } else if (trigName.find("SC", etStrPos + 1) < trigName.find("EG", etStrPos + 1) &&
               trigName.find("SC", etStrPos + 1) < trigName.find("Ele", etStrPos + 1) &&
               trigName.find("SC", etStrPos + 1) < trigName.find("Photon", etStrPos + 1)) {
      etStrPos = trigName.find("Photon", etStrPos + 1);
      isSC2 = true;
    }
    //std::cout<<"Got second Et spot; etStrPos="<<etStrPos<<std::endl;//}//get second instance.  if it dne, keep first

    if (isEle2) {
      if (etStrPos != trigName.npos &&
          trigName.find_first_of("1234567890", etStrPos) == etStrPos + 3) {  //std::cout<<"In if"<<std::endl;
        size_t endOfEtValStr = trigName.find_first_not_of("1234567890", etStrPos + 3);

        std::istringstream etValStr(trigName.substr(etStrPos + 3, endOfEtValStr - etStrPos - 3));
        float etVal;
        etValStr >> etVal;  //std::cout<<"TrigName= "<<trigName<<"   etVal= "<<etVal<<std::endl;
        return etVal;
      }
    }
    if (isEG2 || isSC2) {
      if (etStrPos != trigName.npos &&
          trigName.find_first_of("1234567890", etStrPos) == etStrPos + 2) {  //std::cout<<"In if"<<std::endl;
        size_t endOfEtValStr = trigName.find_first_not_of("1234567890", etStrPos + 2);

        std::istringstream etValStr(trigName.substr(etStrPos + 2, endOfEtValStr - etStrPos - 2));
        float etVal;
        etValStr >> etVal;  //std::cout<<"TrigName= "<<trigName<<"   etVal= "<<etVal<<std::endl;
        return etVal;
      }
    }

    if (isPhoton2) {
      if (etStrPos != trigName.npos &&
          trigName.find_first_of("1234567890", etStrPos) == etStrPos + 6) {  //std::cout<<"In if"<<std::endl;
        size_t endOfEtValStr = trigName.find_first_not_of("1234567890", etStrPos + 6);

        std::istringstream etValStr(trigName.substr(etStrPos + 6, endOfEtValStr - etStrPos - 6));
        float etVal;
        etValStr >> etVal;  //std::cout<<"TrigName= "<<trigName<<"   etVal= "<<etVal<<std::endl;
        return etVal;
      }
    }
  } else if (etStrPos != trigName.npos) {
    if (isEle) {
      if (etStrPos != trigName.npos &&
          trigName.find_first_of("1234567890", etStrPos) == etStrPos + 3) {  //std::cout<<"In if"<<std::endl;
        size_t endOfEtValStr = trigName.find_first_not_of("1234567890", etStrPos + 3);

        std::istringstream etValStr(trigName.substr(etStrPos + 3, endOfEtValStr - etStrPos - 3));
        float etVal;
        etValStr >> etVal;  //std::cout<<"TrigName= "<<trigName<<"   etVal= "<<etVal<<std::endl;
        return etVal;
      }
    }
    if (isEG) {
      if (etStrPos != trigName.npos &&
          trigName.find_first_of("1234567890", etStrPos) == etStrPos + 2) {  //std::cout<<"In if"<<std::endl;
        size_t endOfEtValStr = trigName.find_first_not_of("1234567890", etStrPos + 2);

        std::istringstream etValStr(trigName.substr(etStrPos + 2, endOfEtValStr - etStrPos - 2));
        float etVal;
        etValStr >> etVal;  //std::cout<<"TrigName= "<<trigName<<"   etVal= "<<etVal<<std::endl;
        return etVal;
      }
    }

    if (isPhoton) {
      if (etStrPos != trigName.npos &&
          trigName.find_first_of("1234567890", etStrPos) == etStrPos + 6) {  //std::cout<<"In if"<<std::endl;
        size_t endOfEtValStr = trigName.find_first_not_of("1234567890", etStrPos + 6);

        std::istringstream etValStr(trigName.substr(etStrPos + 6, endOfEtValStr - etStrPos - 6));
        float etVal;
        etValStr >> etVal;  //std::cout<<"TrigName= "<<trigName<<"   etVal= "<<etVal<<std::endl;
        return etVal;
      }
    }
  }

  return 0;
}
