#include <iostream>
#include <string>
#include "GeneratorInterface/HiGenCommon/interface/BaseHiGenEvtSelector.h"

class HiGenEvtSelectorFactory {
 public: 
   HiGenEvtSelectorFactory(){;}
   virtual ~HiGenEvtSelectorFactory(){;}
   
   static BaseHiGenEvtSelector* get(std::string, const edm::ParameterSet&);
   
};

