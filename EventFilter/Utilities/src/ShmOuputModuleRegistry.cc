#include "EventFilter/Utilities/interface/ShmOutputModuleRegistry.h"
#include "EventFilter/Modules/src/FUShmOutputModule.h"


namespace evf{

  ShmOutputModuleRegistry::ShmOutputModuleRegistry(const edm::ParameterSet &ps){
  }

  void ShmOutputModuleRegistry::registerModule(std::string &name, edm::FUShmOutputModule *op)
  {
    clm_.insert(std::pair<std::string, edm::FUShmOutputModule*>(name,op));
  }
  

  edm::FUShmOutputModule* ShmOutputModuleRegistry::get(std::string &name)
  {
    edm::FUShmOutputModule* retval = 0;
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
