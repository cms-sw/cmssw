#ifndef Fireworks_Core_FWBoxIconBase_h
#define Fireworks_Core_FWBoxIconBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWBoxIconBase
// 
/**\class FWBoxIconBase FWBoxIconBase.h Fireworks/Core/interface/FWBoxIconBase.h

 Description: Base class for rendering an icon which has a box as an outline

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 19 15:09:30 CST 2009
// $Id: FWBoxIconBase.h,v 1.1 2009/03/04 16:40:50 chrjones Exp $
//

// system include files
#include "GuiTypes.h"

// user include files

// forward declarations

class FWBoxIconBase {
   
public:
   FWBoxIconBase(unsigned int iEdgeLength);
   virtual ~FWBoxIconBase();
   
   // ---------- const member functions ---------------------
   void draw(Drawable_t iID, GContext_t iContext, int iX, int iY) const;
   
   unsigned int edgeLength() const { return m_edgeLength;}
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   
private:
   FWBoxIconBase(const FWBoxIconBase&); // stop default
   
   const FWBoxIconBase& operator=(const FWBoxIconBase&); // stop default

   virtual void drawInsideBox(Drawable_t iID, GContext_t iContext, int iX, int iY, unsigned int iSize) const= 0;
   
   // ---------- member data --------------------------------
   unsigned int m_edgeLength;
};


#endif
