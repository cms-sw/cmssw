#ifndef __PhysicsTools_PatAlgos_ObjectModifier_h__
#define __PhysicsTools_PatAlgos_ObjectModifier_h__

#include "PhysicsTools/PatAlgos/interface/ModifyObjectValueBase.h"
#include <memory>

namespace pat {
  template<class T>
  class ObjectModifier {
  public:
    typedef std::unique_ptr<ModifyObjectValueBase> ModifierPointer;

    ObjectModifier(const edm::ParameterSet& conf);
    ~ObjectModifier() {}

    void setEvent(const edm::EventBase& event) {
      for( unsigned i = 0; i < modifiers_.size(); ++i )
        modifiers_[i]->setEvent(event);
    }
    
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    void setConsumes(edm::ConsumesCollector& sumes) {
      for( unsigned i = 0; i < modifiers_.size(); ++i )
        modifiers_[i]->setConsumes(sumes);
    }
#endif
    
    void modify(T& obj) const {
      for( unsigned i = 0; i < modifiers_.size(); ++i ) 
        modifiers_[i]->modifyObject(obj);
    }

  private:
    std::vector<ModifierPointer> modifiers_;
  };

  template<class T>
  ObjectModifier<T>::ObjectModifier(const edm::ParameterSet& conf) {
    const std::vector<edm::ParameterSet>& mods = 
      conf.getParameterSetVector("modifications");
    for(unsigned i = 0; i < mods.size(); ++i ) {
      const edm::ParameterSet& iconf = mods[i];
      const std::string& mname = iconf.getParameter<std::string>("name");
      ModifyObjectValueBase* plugin = nullptr;
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
      plugin = ModifyObjectValueFactory::get()->create(mname,iconf);
#endif
      if( nullptr != plugin ) {
        modifiers_.push_back(ModifierPointer(plugin));
      } else {
        throw cms::Exception("BadPluginName")
          << "The requested modifier: " << mname << " is not available!";
      }
    }
  }
}

#endif
