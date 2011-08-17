#ifndef EVF_SHMOUTPUTMODULE_REGISTRY_H
#define EVF_SHMOUTPUTMODULE_REGISTRY_H

#include <string>
#include <map>

namespace edm{
  class ParameterSet;
}



namespace evf
{
  
  class OutputModule{
  public: 
    unsigned int getCounts()=0;
  };
  class ShmOutputModuleRegistry
    {
    public:
      ShmOutputModuleRegistry(const edm::ParameterSet &);
      OutputModule *get(std::string &name);
      void registerModule(std::string &name, OutputModule *op);
      void dumpRegistry();

    private:
      typedef std::map<std::string, edm::FUShmOutputModule*> dct;
      typedef dct::iterator idct;
      void clear();
      dct clm_;
      friend class FWEPWrapper;
    };
}
#endif

