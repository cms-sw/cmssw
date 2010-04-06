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
// $Id$
//

// system include files

// user include files
#include "Rtypes.h"
#include "Fireworks/Core/interface/FWViewType.h"

// forward declarations
class TGeoHMatrix;
class TGeoShape;

class TEveElement;
class TEveElementList;
class TEveGeoShape;

class DetIdToMatrix;
class FWColorManager;

class FWRPZViewGeometry
{
public:
   FWRPZViewGeometry(const DetIdToMatrix*,  const FWColorManager*);
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
   TEveElement* makeTrackerGeometryRhoPhi();
   TEveElement* makeTrackerGeometryRhoZ();
   void estimateProjectionSizeDT(const TGeoHMatrix*, const TGeoShape*, double&, double&, double&, double& );
   void estimateProjectionSizeCSC(const TGeoHMatrix*, const TGeoShape*, double&, double&, double&, double& );
   void estimateProjectionSize( const Double_t*, double&, double&, double&, double& );

   TEveGeoShape* makeShape( const char*, double, double, double, double );

   const DetIdToMatrix*    m_detIdToMatrix;
   const FWColorManager*   m_colorManager;

   static TEveElementList*  s_rhoPhiGeo;
   static TEveElementList*  s_rhoZGeo;
};


#endif
