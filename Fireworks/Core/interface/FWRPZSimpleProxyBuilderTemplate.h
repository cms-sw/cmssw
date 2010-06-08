#ifndef Fireworks_Core_FWRPZSimpleProxyBuilderTemplate_h
#define Fireworks_Core_FWRPZSimpleProxyBuilderTemplate_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZSimpleProxyBuilderTemplate
//
/**\class FWRPZSimpleProxyBuilderTemplate FWRPZSimpleProxyBuilderTemplate.h Fireworks/Core/interface/FWRPZSimpleProxyBuilderTemplate.h

   Description:Base class for proxy builder that creates a graphical object based on an instance of class T

   Usage:
    override the method

   void build(const T& iData, unsigned int iIndex, TEveElement& oItemHolder) const

   where
    iData : the data from the Event which you are supposed to make a graphical representation
    iIndex: the index in the container which iData was obtained
    oItemHolder: a TEveElement which holds the object(s) that inherit from TEveElement which you create
   to form the graphical representation of iData.  Add your TEveElements to iItemHolder by calling the
   method AddElement(TEveElement*)


 */
//
// Original Author:  Chris Jones
//         Created:  Sat Nov 22 10:42:35 CST 2008
// $Id: FWRPZSimpleProxyBuilderTemplate.h,v 1.1 2008/11/26 02:15:49 chrjones Exp $
//

// system include files

// user include files

// forward declarations
#include "Fireworks/Core/interface/FWRPZSimpleProxyBuilder.h"

template <typename T>
class FWRPZSimpleProxyBuilderTemplate : public FWRPZSimpleProxyBuilder {

public:
   FWRPZSimpleProxyBuilderTemplate() :
      FWRPZSimpleProxyBuilder(typeid(T)) {
   }
   //virtual ~FWRPZSimpleProxyBuilderTemplate();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FWRPZSimpleProxyBuilderTemplate(const FWRPZSimpleProxyBuilderTemplate&); // stop default

   const FWRPZSimpleProxyBuilderTemplate& operator=(const FWRPZSimpleProxyBuilderTemplate&); // stop default

   // ---------- member data --------------------------------
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

};


#endif
