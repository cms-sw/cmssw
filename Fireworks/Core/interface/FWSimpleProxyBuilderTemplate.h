#ifndef Fireworks_Core_FWSimpleProxyBuilderTemplate_h
#define Fireworks_Core_FWSimpleProxyBuilderTemplate_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWSimpleProxyBuilderTemplate
//
/**\class FWSimpleProxyBuilderTemplate FWSimpleProxyBuilderTemplate.h Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 11:20:00 EST 2008
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"

// forward declarations

template <typename T>
class FWSimpleProxyBuilderTemplate : public FWSimpleProxyBuilder {
public:
  FWSimpleProxyBuilderTemplate() : FWSimpleProxyBuilder(typeid(T)) {}

  //virtual ~FWSimpleProxyBuilderTemplate();

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

protected:
  const T& modelData(int index) { return *reinterpret_cast<const T*>(m_helper.offsetObject(item()->modelData(index))); }

  using FWSimpleProxyBuilder::build;
  void build(const void* iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* context) override {
    if (nullptr != iData) {
      build(*reinterpret_cast<const T*>(iData), iIndex, oItemHolder, context);
    }
  }

  using FWSimpleProxyBuilder::buildViewType;
  void buildViewType(const void* iData,
                     unsigned int iIndex,
                     TEveElement& oItemHolder,
                     FWViewType::EType viewType,
                     const FWViewContext* context) override {
    if (nullptr != iData) {
      buildViewType(*reinterpret_cast<const T*>(iData), iIndex, oItemHolder, viewType, context);
    }
  }
  /**iIndex is the index where iData is found in the container from which it came
      iItemHolder is the object to which you add your own objects which inherit from TEveElement
   */
  virtual void build(const T& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) {
    throw std::runtime_error(
        "virtual build(const T&, unsigned int, TEveElement&, const FWViewContext*) not implemented by inherited "
        "class.");
  }

  virtual void buildViewType(
      const T& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType viewType, const FWViewContext*) {
    throw std::runtime_error(
        "virtual buildViewType(const T&, unsigned int, TEveElement&, FWViewType::EType, const FWViewContext*) not "
        "implemented by inherited class");
  };

private:
  FWSimpleProxyBuilderTemplate(const FWSimpleProxyBuilderTemplate&) = delete;  // stop default

  const FWSimpleProxyBuilderTemplate& operator=(const FWSimpleProxyBuilderTemplate&) = delete;  // stop default

  // ---------- member data --------------------------------
};

#endif
