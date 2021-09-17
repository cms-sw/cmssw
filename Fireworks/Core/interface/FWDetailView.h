// -*- C++ -*-
#ifndef Fireworks_Core_FWDetailView_h
#define Fireworks_Core_FWDetailView_h

#include "FWCore/Reflection/interface/TypeWithDict.h"
#include <string>
#include <typeinfo>
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Core/interface/FWDetailViewFactory.h"

template <typename T>
class FWDetailView : public FWDetailViewBase {
public:
  FWDetailView() : FWDetailViewBase(typeid(T)) {}

  static std::string classTypeName() { return edm::TypeWithDict(typeid(T)).name(); }

  static std::string classRegisterTypeName() { return typeid(T).name(); }
  virtual void build(const FWModelId&, const T*) = 0;
  virtual void setTextInfo(const FWModelId&, const T*) = 0;

  void build(const FWModelId& iID, const void* iData) override {
    setItem(iID.item());
    build(iID, reinterpret_cast<const T*>(iData));
  }
};

#endif
