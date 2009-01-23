#ifndef Fireworks_Core_FWEveValueScaler_h
#define Fireworks_Core_FWEveValueScaler_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveValueScaler
//
/**\class FWEveValueScaler FWEveValueScaler.h Fireworks/Core/interface/FWEveValueScaler.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Wed Jul  2 15:48:26 EDT 2008
// $Id: FWEveValueScaler.h,v 1.2 2008/11/06 22:05:22 amraktad Exp $
//

// system include files
#include "TEveElement.h"

// user include files
class FWEveValueScaled;

// forward declarations

class FWEveValueScaler
{

public:
   FWEveValueScaler(float iScale=1.) : m_scale(iScale) {
   }
   virtual ~FWEveValueScaler();

   // ---------- const member functions ---------------------
   float scale() const {
      return m_scale;
   }

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   //the object must also inherit from FWEveValueScaled
   void addElement(TEveElement* iElement);
   void removeElement(TEveElement* iElement);
   void removeElements();

   void setScale(float);
private:
   FWEveValueScaler(const FWEveValueScaler&);    // stop default

   const FWEveValueScaler& operator=(const FWEveValueScaler&);    // stop default

   // ---------- member data --------------------------------
   TEveElementList m_scalables;
   float m_scale;
};


#endif
