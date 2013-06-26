#ifndef Fireworks_Core_FWColorBoxIcon_h
#define Fireworks_Core_FWColorBoxIcon_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWColorBoxIcon
// 
/**\class FWColorBoxIcon FWColorBoxIcon.h Fireworks/Core/interface/FWColorBoxIcon.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 19 15:42:03 CST 2009
// $Id: FWColorBoxIcon.h,v 1.1 2009/03/04 16:40:50 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/src/FWBoxIconBase.h"

// forward declarations

class FWColorBoxIcon : public FWBoxIconBase {
   
public:
   FWColorBoxIcon(unsigned int iEdgeLength);
   //virtual ~FWColorBoxIcon();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void setColor(GContext_t iColorContext)
   {
      m_colorContext = iColorContext;
   }
   
private:
   FWColorBoxIcon(const FWColorBoxIcon&); // stop default
   
   const FWColorBoxIcon& operator=(const FWColorBoxIcon&); // stop default

   void drawInsideBox(Drawable_t iID, GContext_t iContext, int iX, int iY, unsigned int iSize) const;

   // ---------- member data --------------------------------
   GContext_t m_colorContext ;
};


#endif
