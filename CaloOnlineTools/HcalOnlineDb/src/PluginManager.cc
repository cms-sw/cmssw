#include "CaloOnlineTools/HcalOnlineDb/interface/PluginManager.hh"

namespace hcal {

  std::map<std::string, std::map<std::string, AbstractPluginFactory*> > PluginManager::m_factories;

  void PluginManager::registerFactory(const char* baseClass, const char* derivedClass, AbstractPluginFactory* factory) {
    m_factories[baseClass][derivedClass]=factory;
  }

  void PluginManager::getFactories(const char* baseClass, std::vector<AbstractPluginFactory*>& factories) {
    factories.clear();
    std::map<std::string, std::map<std::string, AbstractPluginFactory*> >::const_iterator j=m_factories.find(baseClass);
    if (j==m_factories.end()) return;
    std::map<std::string, AbstractPluginFactory*>::const_iterator i;
    for (i=j->second.begin(); i!=j->second.end(); i++) 
      factories.push_back(i->second);
  }

  AbstractPluginFactory* PluginManager::getFactory(const char* baseClass, const char* derivedClass) {
    std::map<std::string, std::map<std::string, AbstractPluginFactory*> >::const_iterator j=m_factories.find(baseClass);
    if (j==m_factories.end()) return 0;
    std::map<std::string, AbstractPluginFactory*>::const_iterator i=j->second.find(derivedClass);
    if (i==j->second.end()) return 0;
    return i->second;
  }

}
