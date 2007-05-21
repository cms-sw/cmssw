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
// $Id: PluginFactory.h,v 1.4 2007/04/17 22:15:45 wmtan Exp $
//

// system include files
#include <map>
#include <vector>

// user include files
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
// forward declarations

namespace edmplugin {
  template< class T> class PluginFactory;
  class DummyFriend;
  
template<class R>
class PluginFactory<R * (void)> : public PluginFactoryBase
{
      friend class DummyFriend;
   public:
      struct PMakerBase {
        virtual R* create(void) const = 0;
        virtual ~PMakerBase() {}
      };
      template<class TPlug>
      struct PMaker : public PMakerBase {
        PMaker(const std::string& iName) {
          PluginFactory<R*(void)>::get()->registerPMaker(this,iName);
        }
        virtual R* create() const {
          return new TPlug();
        }
      };
      typedef std::vector<std::pair<PMakerBase*, std::string> > PMakers;
      typedef std::map<std::string, PMakers > Plugins;

      // ---------- const member functions ---------------------
      virtual std::vector<PluginInfo> available() const {
        std::vector<PluginInfo> returnValue;
        returnValue.reserve(m_plugins.size());
        fillAvailable(m_plugins.begin(),
                      m_plugins.end(),
                      returnValue);
        return returnValue;
      }
      virtual const std::string& category() const ;
      
      R* create(const std::string& iName) const {
        return PluginFactoryBase::findPMaker(iName,m_plugins)->second.front().first->create();
      }
      // ---------- static member functions --------------------

      static PluginFactory<R*(void)>* get();
      // ---------- member functions ---------------------------
      void registerPMaker(PMakerBase* iPMaker, const std::string& iName) {
        m_plugins[iName].push_back(std::pair<PMakerBase*,std::string>(iPMaker,PluginManager::loadingFile()));
        newPlugin(iName);
      }

   private:
      PluginFactory() {
        finishedConstruction();
      }
      PluginFactory(const PluginFactory&); // stop default

      const PluginFactory& operator=(const PluginFactory&); // stop default

      // ---------- member data --------------------------------
      Plugins m_plugins;

};

template<class R, class Arg>
class PluginFactory<R * (Arg)> : public PluginFactoryBase
{
  friend class DummyFriend;
public:
  struct PMakerBase {
    virtual R* create(Arg) const = 0;
    virtual ~PMakerBase() {}
  };
  template<class TPlug>
    struct PMaker : public PMakerBase {
      PMaker(const std::string& iName) {
        PluginFactory<R*(Arg)>::get()->registerPMaker(this,iName);
      }
      virtual R* create(Arg iArg) const {
        return new TPlug(iArg);
      }
    };
  typedef std::vector<std::pair<PMakerBase*, std::string> > PMakers;
  typedef std::map<std::string, PMakers > Plugins;
  
  // ---------- const member functions ---------------------
  virtual std::vector<PluginInfo> available() const {
    std::vector<PluginInfo> returnValue;
    returnValue.reserve(m_plugins.size());
    fillAvailable(m_plugins.begin(),
                  m_plugins.end(),
                  returnValue);
    return returnValue;
  }
  virtual const std::string& category() const ;
  
  R* create(const std::string& iName, Arg iArg) const {
    return PluginFactoryBase::findPMaker(iName,m_plugins)->second.front().first->create(iArg);
  }
  // ---------- static member functions --------------------
  
  static PluginFactory<R*(Arg)>* get();
  // ---------- member functions ---------------------------
  void registerPMaker(PMakerBase* iPMaker, const std::string& iName) {
    m_plugins[iName].push_back(std::pair<PMakerBase*,std::string>(iPMaker,PluginManager::loadingFile()));
    newPlugin(iName);
  }
  
private:
    PluginFactory() {
      finishedConstruction();
    }
  PluginFactory(const PluginFactory&); // stop default
  
  const PluginFactory& operator=(const PluginFactory&); // stop default
  
  // ---------- member data --------------------------------
  Plugins m_plugins;
  
};

template<class R, class Arg1, class Arg2>
class PluginFactory<R * (Arg1, Arg2)> : public PluginFactoryBase
{
  friend class DummyFriend;
public:
  struct PMakerBase {
    virtual R* create(Arg1, Arg2) const = 0;
    virtual ~PMakerBase() {}
  };
  template<class TPlug>
    struct PMaker : public PMakerBase {
      PMaker(const std::string& iName) {
        PluginFactory<R*(Arg1,Arg2)>::get()->registerPMaker(this,iName);
      }
      virtual R* create(Arg1 iArg1, Arg2 iArg2) const {
        return new TPlug(iArg1, iArg2);
      }
    };
  typedef std::vector<std::pair<PMakerBase*, std::string> > PMakers;
  typedef std::map<std::string, PMakers > Plugins;
  
  // ---------- const member functions ---------------------
  virtual std::vector<PluginInfo> available() const {
    std::vector<PluginInfo> returnValue;
    returnValue.reserve(m_plugins.size());
    fillAvailable(m_plugins.begin(),
                  m_plugins.end(),
                  returnValue);
    return returnValue;
  }
  virtual const std::string& category() const ;
  
  R* create(const std::string& iName, Arg1 iArg1, Arg2 iArg2) const {
    return PluginFactoryBase::findPMaker(iName,m_plugins)->second.front().first->create(iArg1, iArg2);
  }
  // ---------- static member functions --------------------
  
  static PluginFactory<R*(Arg1,Arg2)>* get();
  // ---------- member functions ---------------------------
  void registerPMaker(PMakerBase* iPMaker, const std::string& iName) {
    m_plugins[iName].push_back(std::pair<PMakerBase*,std::string>(iPMaker,PluginManager::loadingFile()));
    newPlugin(iName);
  }
  
private:
    PluginFactory() {
      finishedConstruction();
    }
  PluginFactory(const PluginFactory&); // stop default
  
  const PluginFactory& operator=(const PluginFactory&); // stop default
  
  // ---------- member data --------------------------------
  Plugins m_plugins;
};

}
#define CONCATENATE_HIDDEN(a,b) a ## b 
#define CONCATENATE(a,b) CONCATENATE_HIDDEN(a,b)
#define EDM_REGISTER_PLUGINFACTORY(_factory_,_category_) \
namespace edmplugin {\
  template<> _factory_* _factory_::get() { static _factory_ s_instance; return &s_instance;}\
  template<> const std::string& _factory_::category() const { static std::string s_cat(_category_);  return s_cat;}\
} enum {CONCATENATE(dummy_edm_register_pluginfactory_, __LINE__)}

#endif

#define EDM_PLUGIN_SYM(x,y) EDM_PLUGIN_SYM2(x,y)
#define EDM_PLUGIN_SYM2(x,y) x ## y

#define DEFINE_EDM_PLUGIN(factory,type,name) \
static factory::PMaker<type> EDM_PLUGIN_SYM(s_maker , __LINE__ ) (name)

//for backwards compatiblity
#include "FWCore/PluginManager/interface/ModuleDef.h"
