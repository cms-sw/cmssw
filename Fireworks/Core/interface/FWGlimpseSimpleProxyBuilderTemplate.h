#ifndef Fireworks_Core_FWGlimpseSimpleProxyBuilderTemplate_h
#define Fireworks_Core_FWGlimpseSimpleProxyBuilderTemplate_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseSimpleProxyBuilderTemplate
//
/**\class FWGlimpseSimpleProxyBuilderTemplate FWGlimpseSimpleProxyBuilderTemplate.h Fireworks/Core/interface/FWGlimpseSimpleProxyBuilderTemplate.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 11:20:00 EST 2008
// $Id: FWGlimpseSimpleProxyBuilderTemplate.h,v 1.1 2008/12/02 21:11:52 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWGlimpseSimpleProxyBuilder.h"

// forward declarations

template <typename T>
class FWGlimpseSimpleProxyBuilderTemplate : public FWGlimpseSimpleProxyBuilder {

public:
   FWGlimpseSimpleProxyBuilderTemplate() :
      FWGlimpseSimpleProxyBuilder(typeid(T)) {
   }

   //virtual ~FWGlimpseSimpleProxyBuilderTemplate();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWGlimpseSimpleProxyBuilderTemplate(const FWGlimpseSimpleProxyBuilderTemplate&); // stop default

   const FWGlimpseSimpleProxyBuilderTemplate& operator=(const FWGlimpseSimpleProxyBuilderTemplate&); // stop default

   virtual void build(const void*iData, unsigned int iIndex, TEveElement& oItemHolder) const
   {
      if(0!=iData) {
         build(*reinterpret_cast<const T*> (iData), iIndex, oItemHolder);
      }
   }

   /**iIndex is the index where iData is found in the container from which it came
      iItemHolder is the object to which you add your own objects which inherit from TEveElement
    */
   virtual void build(const T& iData, unsigned int iIndex,TEveElement& oItemHolder) const = 0;

   // ---------- member data --------------------------------

};


#endif
