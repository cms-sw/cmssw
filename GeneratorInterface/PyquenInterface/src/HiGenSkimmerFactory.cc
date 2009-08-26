
#include "GeneratorInterface/PyquenInterface/interface/HiGenSkimmerFactory.h"
#include "GeneratorInterface/PyquenInterface/interface/EcalCandidateSkimmer.h"
#include "GeneratorInterface/PyquenInterface/interface/MultipleCandidateSkimmer.h"

BaseHiGenSkimmer* HiGenSkimmerFactory::get(std::string filterType, const edm::ParameterSet& pset){
   if(filterType == "None"){
      return new BaseHiGenSkimmer(pset);
   }else if(filterType == "EcalCandidateSkimmer"){
      return new EcalCandidateSkimmer(pset);
   }else if(filterType == "MultipleCandidateSkimmer"){
      return new MultipleCandidateSkimmer(pset);
   }

   std::cout<<"Skimmer not recognized. Fail!"<<std::endl;
   return NULL;
}


