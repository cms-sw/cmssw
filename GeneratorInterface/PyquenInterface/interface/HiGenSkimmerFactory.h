#include <iostream>
#include <string>
#include "GeneratorInterface/PyquenInterface/interface/BaseHiGenSkimmer.h"

class HiGenSkimmerFactory {
 public: 
   HiGenSkimmerFactory(){;}
   virtual ~HiGenSkimmerFactory(){;}
   
   static BaseHiGenSkimmer* get(std::string, const edm::ParameterSet&);
   
};

