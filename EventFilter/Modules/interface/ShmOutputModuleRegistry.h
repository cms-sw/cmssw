#ifndef EVF_SHMOUTPUTMODULE_REGISTRY_H
#define EVF_SHMOUTPUTMODULE_REGISTRY_H

#include <string>
#include <map>
#include <vector>

namespace edm{
  class ParameterSet;
  class FUShmOutputModule;
}



namespace evf
{
  
  class OutputModule{
  public: 
    virtual unsigned int getCounts()=0;
  };
  class ShmOutputModuleRegistry
    {
    public:
      ShmOutputModuleRegistry(const edm::ParameterSet &);
      OutputModule *get(std::string &name);
      void registerModule(std::string &name, OutputModule *op);
      void dumpRegistry();
      std::vector<edm::FUShmOutputModule *> getShmOutputModules();

      std::string getDatasetNamesString();
      std::vector<edm::FUShmOutputModule *> & getShmOutputModulesWithDatasets() {return shmOutputsWithDatasets_;}

    private:
      typedef std::map<std::string, OutputModule*> dct;
      typedef dct::iterator idct;
      void clear();
      dct clm_;
      std::vector<edm::FUShmOutputModule *> shmOutputsWithDatasets_;
      friend class FWEPWrapper;
    };
}
#endif

