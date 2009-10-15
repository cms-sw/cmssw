
#include "GeneratorInterface/HiGenCommon/interface/HiGenEvtSelectorFactory.h"
#include "GeneratorInterface/HiGenCommon/interface/EcalGenEvtSelector.h"
#include "GeneratorInterface/HiGenCommon/interface/MultiCandGenEvtSelector.h"

BaseHiGenEvtSelector* HiGenEvtSelectorFactory::get(std::string filterType, const edm::ParameterSet& pset){
   if(filterType == "None"){
      return new BaseHiGenEvtSelector(pset);
   }else if(filterType == "EcalGenEvtSelector"){
      return new EcalGenEvtSelector(pset);
   }else if(filterType == "MultiCandGenEvtSelector"){
      return new MultiCandGenEvtSelector(pset);
   }

   std::cout<<"Skimmer not recognized. Fail!"<<std::endl;
   return NULL;
}


