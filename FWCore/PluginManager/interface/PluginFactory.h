#ifndef FWCore_PluginManager_PluginFactory_h
#define FWCore_PluginManager_PluginFactory_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginFactory
// 
/**\class PluginFactory PluginFactory.h FWCore/PluginManager/interface/PluginFactory.h

 Description: Public interface for loading a plugin

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Apr  5 12:10:23 EDT 2007
//

// system include files
#include <map>
#include <vector>

// user include files
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
// forward declarations

namespace edmplugin {
template< class T> class PluginFactory;
  class DummyFriend;
  
template<typename R, typename... Args>
class PluginFactory<R*(Args...)> : public PluginFactoryBase
{
      friend class DummyFriend;
   public:
      typedef R* TemplateArgType(Args...);

      struct PMakerBase {
        virtual R* create(Args&&...) const = 0;
        virtual ~PMakerBase() {}
      };
      template<class TPlug>
      struct PMaker : public PMakerBase {
        PMaker(const std::string& iName) {
          PluginFactory<R*(Args...)>::get()->registerPMaker(this,iName);
        }
        virtual R* create(Args&&... args) const {
          return new TPlug(std::forward<Args>(args)...);
        }
      };

      // ---------- const member functions ---------------------
      virtual const std::string& category() const ;
      
      R* create(const std::string& iName, Args&&... args) const {
        return reinterpret_cast<PMakerBase*>(PluginFactoryBase::findPMaker(iName)->second.front().first)->create(std::forward<Args>(args)...);
      }

      ///like above but returns 0 if iName is unknown
      R* tryToCreate(const std::string& iName, Args&&... args) const {
        typename Plugins::const_iterator itFound = PluginFactoryBase::tryToFindPMaker(iName);
        if(itFound ==m_plugins.end() ) {
          return 0;
        }
        return reinterpret_cast<PMakerBase*>(itFound->second.front().first)->create(std::forward<Args>(args)...);
      }
      // ---------- static member functions --------------------

      static PluginFactory<R*(Args...)>* get();
      // ---------- member functions ---------------------------
      void registerPMaker(PMakerBase* iPMaker, const std::string& iName) {
        PluginFactoryBase::registerPMaker(iPMaker, iName);
      }

   private:
      PluginFactory() {
        finishedConstruction();
      }
      PluginFactory(const PluginFactory&) = delete; // stop default

      const PluginFactory& operator=(const PluginFactory&) = delete; // stop default

};
}
#define CONCATENATE_HIDDEN(a,b) a ## b 
#define CONCATENATE(a,b) CONCATENATE_HIDDEN(a,b)
#define EDM_REGISTER_PLUGINFACTORY(_factory_,_category_) \
namespace edmplugin {\
  template<> edmplugin::PluginFactory<_factory_::TemplateArgType>* edmplugin::PluginFactory<_factory_::TemplateArgType>::get() { static edmplugin::PluginFactory<_factory_::TemplateArgType> s_instance; return &s_instance;}\
  template<> const std::string& edmplugin::PluginFactory<_factory_::TemplateArgType>::category() const { static std::string s_cat(_category_);  return s_cat;}\
  } enum {CONCATENATE(dummy_edm_register_pluginfactory_, __LINE__)}

#endif

#define EDM_PLUGIN_SYM(x,y) EDM_PLUGIN_SYM2(x,y)
#define EDM_PLUGIN_SYM2(x,y) x ## y

#define DEFINE_EDM_PLUGIN(factory,type,name) \
static factory::PMaker<type> EDM_PLUGIN_SYM(s_maker , __LINE__ ) (name)

