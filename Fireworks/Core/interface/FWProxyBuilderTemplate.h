#ifndef Fireworks_Core_FWProxyBuilderTemplate_h
#define Fireworks_Core_FWProxyBuilderTemplate_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWProxyBuilderTemplate
//
/**\class FWProxyBuilderTemplate FWProxyBuilderTemplate.h Fireworks/Core/interface/FWProxyBuilderTemplate.h

   Description: <one line class summary>

   Usage:s
    <usage>

 */
//
// Original Author:  Matevz Tadel
//         Created:  April 23 2010
//

// system include files
#include <typeinfo>

// user include files
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"
#include "Fireworks/Core/interface/FWEventItem.h"

template <typename T>
class FWProxyBuilderTemplate : public FWProxyBuilderBase
{
public:
   FWProxyBuilderTemplate() : m_helper(typeid(T)) {}
   ~FWProxyBuilderTemplate() override {}

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

protected:
   const T& modelData(int index) { return *reinterpret_cast<const T*>(m_helper.offsetObject(item()->modelData(index))); }

private:
   FWProxyBuilderTemplate(const FWProxyBuilderTemplate&) = delete; // stop default

   const FWProxyBuilderTemplate& operator=(const FWProxyBuilderTemplate&) = delete; // stop default

   virtual void itemChangedImp(const FWEventItem* iItem) { if (iItem) m_helper.itemChanged(iItem); }
   
   // ---------- member data --------------------------------
   FWSimpleProxyHelper m_helper;
};

#endif
