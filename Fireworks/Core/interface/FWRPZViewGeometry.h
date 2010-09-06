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
// $Id: FWRPZViewGeometry.h,v 1.4 2010/08/31 15:30:19 yana Exp $
//

// system include files

// user include files
#include "Rtypes.h"
#include "Fireworks/Core/interface/FWViewType.h"

// forward declarations
class TGeoMatrix;
class TGeoShape;

class TEveElement;
class TEveElementList;
class TEveGeoShape;

class DetIdToMatrix;
class FWColorManager;

namespace fireworks
{
   class Context;
}

class FWRPZViewGeometry
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
   void estimateProjectionSizeDT( const TGeoMatrix*, const TGeoShape*, double&, double&, double&, double& );
   void estimateProjectionSizeCSC( const TGeoMatrix*, const TGeoShape*, double&, double&, double&, double& );
   void estimateProjectionSize( const Double_t*, double&, double&, double&, double& );

   TEveGeoShape* makeShape( double, double, double, double, Color_t );

   const fireworks::Context&    m_context; // cached
   const DetIdToMatrix*    m_geom; // cached

   static TEveElementList*  s_rhoPhiGeo;
   static TEveElementList*  s_rhoZGeo;
};


#endif
