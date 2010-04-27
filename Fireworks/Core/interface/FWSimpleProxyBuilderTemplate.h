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
// $Id: FWSimpleProxyBuilderTemplate.h,v 1.3 2010/04/23 21:01:59 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilder.h"

// forward declarations

template <typename T>
class FWSimpleProxyBuilderTemplate : public FWSimpleProxyBuilder {

public:
   FWSimpleProxyBuilderTemplate() :
      FWSimpleProxyBuilder(typeid(T)) {
   }

   //virtual ~FWSimpleProxyBuilderTemplate();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

protected:
   const T& modelData(int index) { return *reinterpret_cast<const T*>(m_helper.offsetObject(item()->modelData(index))); }

private:
   FWSimpleProxyBuilderTemplate(const FWSimpleProxyBuilderTemplate&); // stop default

   const FWSimpleProxyBuilderTemplate& operator=(const FWSimpleProxyBuilderTemplate&); // stop default

   virtual void build(const void*iData, unsigned int iIndex, TEveElement& oItemHolder)
   {
      if(0!=iData) {
         build(*reinterpret_cast<const T*> (iData), iIndex, oItemHolder);
      }
   }

   virtual void buildViewType(const void*iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType viewType)
   {
      if(0!=iData) {
         buildViewType(*reinterpret_cast<const T*> (iData), iIndex, oItemHolder, viewType);
      }
   }
   /**iIndex is the index where iData is found in the container from which it came
      iItemHolder is the object to which you add your own objects which inherit from TEveElement
   */
   virtual void build(const T& iData, unsigned int iIndex,TEveElement& oItemHolder) { assert(false);}

   virtual void buildViewType(const T& iData, unsigned int iIndex,TEveElement& oItemHolder, FWViewType::EType viewType) { assert(false);};

   // ---------- member data --------------------------------

};


#endif
