// -*- C++ -*-
#ifndef Fireworks_Core_FWDetailView_h
#define Fireworks_Core_FWDetailView_h

#include "Fireworks/Core/interface/FWDetailViewBase.h"


template<typename T>
class FWDetailView : public FWDetailViewBase {
public:
   FWDetailView() :
   FWDetailViewBase(typeid(T)) {}
private:
   virtual TEveElement* build(const FWModelId& iID, const void* iData) {
      return build(iID, reinterpret_cast<const T*> (iData));
   }
   
   virtual TEveElement* build(const FWModelId&, const T*) = 0;
   
};

#endif
