#include "CaloOnlineTools/HcalOnlineDb/interface/PluginManager.hh"

namespace hcal {

  std::map<std::string, std::map<std::string, AbstractPluginFactory*> > &PluginManager::factories()
  {
    static std::map<std::string, std::map<std::string, AbstractPluginFactory*> > factories;
    return factories;
  }

  void PluginManager::registerFactory(const char* baseClass, const char* derivedClass, AbstractPluginFactory* factory) {
    factories()[baseClass][derivedClass]=factory;
  }

  void PluginManager::getFactories(const char* baseClass, std::vector<AbstractPluginFactory*>& result) {
    result.clear();
    std::map<std::string, std::map<std::string, AbstractPluginFactory*> >::const_iterator j=factories().find(baseClass);
    if (j==factories().end()) return;
    std::map<std::string, AbstractPluginFactory*>::const_iterator i;
    for (i=j->second.begin(); i!=j->second.end(); i++) 
      result.push_back(i->second);
  }

  AbstractPluginFactory* PluginManager::getFactory(const char* baseClass, const char* derivedClass) {
    std::map<std::string, std::map<std::string, AbstractPluginFactory*> >::const_iterator j=factories().find(baseClass);
    if (j==factories().end()) return nullptr;
    std::map<std::string, AbstractPluginFactory*>::const_iterator i=j->second.find(derivedClass);
    if (i==j->second.end()) return nullptr;
    return i->second;
  }

}
