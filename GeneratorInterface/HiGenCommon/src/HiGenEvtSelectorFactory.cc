
#include "GeneratorInterface/HiGenCommon/interface/HiGenEvtSelectorFactory.h"
#include "GeneratorInterface/HiGenCommon/interface/EcalGenEvtSelector.h"
#include "GeneratorInterface/HiGenCommon/interface/EcalGenEvtSelectorFrag.h"
#include "GeneratorInterface/HiGenCommon/interface/MultiCandGenEvtSelector.h"
#include "GeneratorInterface/HiGenCommon/interface/PartonHadronDecayGenEvtSelector.h"
#include "GeneratorInterface/HiGenCommon/interface/HadronDecayGenEvtSelector.h"

BaseHiGenEvtSelector* HiGenEvtSelectorFactory::get(std::string filterType, const edm::ParameterSet& pset){
   if(filterType == "None"){
     return new BaseHiGenEvtSelector(pset);
   }else if(filterType == "EcalGenEvtSelector"){
     return new EcalGenEvtSelector(pset);
   }else if(filterType == "EcalGenEvtSelectorFrag"){
     return new EcalGenEvtSelectorFrag(pset);
   }else if(filterType == "MultiCandGenEvtSelector"){
     return new MultiCandGenEvtSelector(pset);
   }else if(filterType == "PartonHadronDecayGenEvtSelector"){
     return new PartonHadronDecayGenEvtSelector(pset);
   }else if(filterType == "HadronDecayGenEvtSelector"){
     return new HadronDecayGenEvtSelector(pset);
   }

   std::cout<<"Skimmer not recognized. Fail!"<<std::endl;
   return NULL;
}


