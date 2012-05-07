#include "EventFilter/Modules/interface/ShmOutputModuleRegistry.h"
#include "EventFilter/Modules/src/FUShmOutputModule.h"

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
  std::vector<edm::FUShmOutputModule *> ShmOutputModuleRegistry::getShmOutputModules()
  {
    std::vector<edm::FUShmOutputModule *> outputs;
    idct it= clm_.begin();
    while(it!=clm_.end()){
      edm::FUShmOutputModule * sho = dynamic_cast<edm::FUShmOutputModule *> ((*it).second);
      if (sho!=NULL) {
        outputs.push_back(sho);
      }
      it++;
    }
    return outputs;
  }
  void ShmOutputModuleRegistry::clear()
  {
     clm_.clear();
  }


} //end namespace evf
