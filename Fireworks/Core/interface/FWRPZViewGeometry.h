#ifndef Fireworks_Core_FWRPZViewGeometry_h
#define Fireworks_Core_FWRPZViewGeometry_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZViewGeometry
// 
/**\class FWRPZViewGeometry FWRPZViewGeometry.h Fireworks/Core/interface/FWRPZViewGeometry.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Thu Mar 25 21:01:12 CET 2010
// $Id: FWRPZViewGeometry.h,v 1.8 2010/09/15 11:48:42 amraktad Exp $
//

// system include files

// user include files
#include "TEveElement.h"
#include "Fireworks/Core/interface/FWViewType.h"
#include "Fireworks/Core/interface/FWViewGeometryList.h"
#include "Fireworks/Core/interface/FWGeometry.h"

// forward declarations
class TGeoShape;

class FWRPZViewGeometry : public FWViewGeometryList
{
public:
   FWRPZViewGeometry(const fireworks::Context& context);
   virtual ~FWRPZViewGeometry();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   TEveElement* getGeoElements(const FWViewType::EType id);

private:
   FWRPZViewGeometry(const FWRPZViewGeometry&); // stop default
   const FWRPZViewGeometry& operator=(const FWRPZViewGeometry&); // stop default

   // ---------- member data --------------------------------

   TEveElement* makeMuonGeometryRhoPhi();
   TEveElement* makeMuonGeometryRhoZ();
   TEveElement* makeCaloOutlineRhoPhi();
   TEveElement* makeCaloOutlineRhoZ();
   void estimateProjectionSizeDT( const FWGeometry::GeomDetInfo& info, float&, float&, float&, float& );
   void estimateProjectionSizeCSC( const FWGeometry::GeomDetInfo& info, float&, float&, float&, float& );
   void estimateProjectionSize( const float*, float&, float&, float&, float& );

   TEveGeoShape* makeShape( double, double, double, double );

   TEveElementList*  m_rhoPhiGeo;
   TEveElementList*  m_rhoZGeo;
};


#endif
