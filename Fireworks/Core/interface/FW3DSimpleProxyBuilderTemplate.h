#ifndef Fireworks_Core_FW3DSimpleProxyBuilderTemplate_h
#define Fireworks_Core_FW3DSimpleProxyBuilderTemplate_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DSimpleProxyBuilderTemplate
// 
/**\class FW3DSimpleProxyBuilderTemplate FW3DSimpleProxyBuilderTemplate.h Fireworks/Core/interface/FW3DSimpleProxyBuilderTemplate.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 11:20:00 EST 2008
// $Id: FW3DSimpleProxyBuilderTemplate.h,v 1.1 2008/12/02 21:11:52 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DSimpleProxyBuilder.h"

// forward declarations

template <typename T>
class FW3DSimpleProxyBuilderTemplate : public FW3DSimpleProxyBuilder {

public:
   FW3DSimpleProxyBuilderTemplate() :
   FW3DSimpleProxyBuilder(typeid(T)) {}

   //virtual ~FW3DSimpleProxyBuilderTemplate();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   
private:
   FW3DSimpleProxyBuilderTemplate(const FW3DSimpleProxyBuilderTemplate&); // stop default
   
   const FW3DSimpleProxyBuilderTemplate& operator=(const FW3DSimpleProxyBuilderTemplate&); // stop default
   
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
