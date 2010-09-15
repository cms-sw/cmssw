#ifndef Fireworks_Core_FWViewGeometryList_h
#define Fireworks_Core_FWViewGeometryList_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewGeometryList
// 
/**\class FWViewGeometryList FWViewGeometryList.h Fireworks/Core/interface/FWViewGeometryList.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel 
//         Created:  Tue Sep 14 13:28:39 CEST 2010
// $Id$
//

#include "TEveElement.h"
#include "Fireworks/Core/interface/FWColorManager.h"

class TGeoMatrix;
class TEveCompound;
class FWGeometry;

namespace fireworks {
   class Context;
}

class FWViewGeometryList: public TEveElementList
{
public:
   FWViewGeometryList( const fireworks::Context& context );
   virtual ~FWViewGeometryList();

   void updateColors();

protected:
   const fireworks::Context&    m_context;  
   const FWGeometry*            m_geom;    // cached

   TEveCompound*       m_colorComp[kFWGeomColorSize];

   void addToCompound(TEveElement* el, FWGeomColorIndex idx, bool applyTransp = true) const ;

private:
   FWViewGeometryList(const FWViewGeometryList&); // stop default

   const FWViewGeometryList& operator=(const FWViewGeometryList&); // stop default

   // ---------- member data --------------------------------

};


#endif
