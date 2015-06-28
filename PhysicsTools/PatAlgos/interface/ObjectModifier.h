#ifndef __PhysicsTools_PatAlgos_ObjectModifier_h__
#define __PhysicsTools_PatAlgos_ObjectModifier_h__

#include "PhysicsTools/PatAlgos/interface/ModifyObjectValueBase.h"
#include <memory>

namespace pat {
  template<class T>
  class ObjectModifier {
  public:
    ObjectModifier(const edm::ParameterSet& conf);
    ~ObjectModifier() {}

    void setEvent(const edm::EventBase&);
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    void setConsumes(edm::ConsumesCollector&);
#endif
    
    void modify(T& obj) const {
      for( unsigned i = 0; i < modifiers_.size(); ++i ) 
        modifiers_[i]->modifyObject(obj);
    }

  private:
    std::vector<std::unique_ptr<ModifyObjectValueBase> > modifiers_;
  };

  template<class T>
  ObjectModifier<T>::ObjectModifier(const edm::ParameterSet& conf) {
    
  }

  template<class T>
  void ObjectModifier<T>::setEvent(const edm::EventBase&) {
    
  }
  
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  template<class T>
  void ObjectModifier<T>::setConsumes(edm::ConsumesCollector&) {
    
  }
#endif
  
}

#endif
