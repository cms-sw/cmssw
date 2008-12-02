#ifndef Fireworks_Core_FW3DLegoSimpleProxyBuilderTemplate_h
#define Fireworks_Core_FW3DLegoSimpleProxyBuilderTemplate_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoSimpleProxyBuilderTemplate
// 
/**\class FW3DLegoSimpleProxyBuilderTemplate FW3DLegoSimpleProxyBuilderTemplate.h Fireworks/Core/interface/FW3DLegoSimpleProxyBuilderTemplate.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Dec  2 16:49:16 EST 2008
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoSimpleProxyBuilder.h"

// forward declarations

template<typename T>
class FW3DLegoSimpleProxyBuilderTemplate : public FW3DLegoSimpleProxyBuilder {
   
public:
   FW3DLegoSimpleProxyBuilderTemplate() :
   FW3DLegoSimpleProxyBuilder(typeid(T)) {}
   //virtual ~FW3DLegoSimpleProxyBuilderTemplate();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   
private:
   FW3DLegoSimpleProxyBuilderTemplate(const FW3DLegoSimpleProxyBuilderTemplate&); // stop default
   
   const FW3DLegoSimpleProxyBuilderTemplate& operator=(const FW3DLegoSimpleProxyBuilderTemplate&); // stop default

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
