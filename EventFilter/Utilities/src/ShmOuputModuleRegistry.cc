#include "EventFilter/Utilities/interface/ShmOutputModuleRegistry.h"

#include <iostream>

namespace evf{

  ShmOutputModuleRegistry::ShmOutputModuleRegistry(const edm::ParameterSet &ps){
  }

  void ShmOutputModuleRegistry::registerModule(std::string &name, OutputModule *op)
  {
    clm_.insert(std::pair<std::string, OutputModule*>(name,op));
  }
  

  OutputModule* ShmOutputModuleRegistry::get(std::string &name)
  {
    OutputModule* retval = 0;
    idct it= clm_.find(name);
    if(it!=clm_.end())
      retval = (it->second);
    return retval;
  }
  void ShmOutputModuleRegistry::dumpRegistry(){
    idct it= clm_.begin();
    while(it!=clm_.end()){
      std::cout << "name " << it->first << "add " 
		<< (unsigned long)(it->second) << std::endl;
      it++;
    }
  }
  void ShmOutputModuleRegistry::clear()
  {
     clm_.clear();
  }
  

} //end namespace evf
