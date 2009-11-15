// -*- C++ -*-
#ifndef Fireworks_Core_FWDetailView_h
#define Fireworks_Core_FWDetailView_h

#include <Reflex/Type.h>
#include <string>
#include <typeinfo>
#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Core/interface/FWDetailViewFactory.h"


template<typename T>
class FWDetailView : public FWDetailViewBase {
public:
   FWDetailView() :
      FWDetailViewBase(typeid(T)) {
   }

   static std::string classTypeName() {
      return ROOT::Reflex::Type::ByTypeInfo(typeid(T)).Name(ROOT::Reflex::SCOPED);
   }
   
   static std::string classRegisterTypeName() {
      return typeid(T).name();
   }

private:
   virtual TEveElement* build(const FWModelId& iID, const void* iData) {
      return build(iID, reinterpret_cast<const T*> (iData));
   }

   virtual TEveElement* build(const FWModelId&, const T*) = 0;

};

#endif
