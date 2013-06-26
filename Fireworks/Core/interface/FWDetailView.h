// -*- C++ -*-
#ifndef Fireworks_Core_FWDetailView_h
#define Fireworks_Core_FWDetailView_h

#include "FWCore/Utilities/interface/TypeWithDict.h"
#include <string>
#include <typeinfo>
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Core/interface/FWDetailViewFactory.h"


template<typename T>
class FWDetailView : public FWDetailViewBase {
public:
   FWDetailView() :
      FWDetailViewBase(typeid(T)) {
   }

   static std::string classTypeName() {
      return edm::TypeWithDict(typeid(T)).name();
   }

   static std::string classRegisterTypeName() {
      return typeid(T).name();
   }

private:
   virtual void build(const FWModelId& iID, const void* iData) {
      setItem(iID.item());
      build(iID, reinterpret_cast<const T*> (iData));  
   }

   virtual void build(const FWModelId&, const T*) = 0;
   virtual void setTextInfo(const FWModelId&, const T*) = 0;
};

#endif
