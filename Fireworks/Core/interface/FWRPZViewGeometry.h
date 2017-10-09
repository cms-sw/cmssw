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
   void initStdGeoElements(const FWViewType::EType id);

   void showPixelBarrel( bool );
   void showPixelEndcap( bool );
   void showTrackerBarrel( bool );
   void showTrackerEndcap( bool );
   void showRpcEndcap( bool );
   void showGEM( bool );
   void showME0( bool );

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

   void importNew(TEveElementList* x);

   TEveGeoShape* makeShape( double, double, double, double );

   TEveElementList*  m_rhoPhiGeo;
   TEveElementList*  m_rhoZGeo;


   TEveElementList*   m_pixelBarrelElements;
   TEveElementList*   m_pixelEndcapElements;
   TEveElementList*   m_trackerBarrelElements;
   TEveElementList*   m_trackerEndcapElements;
   TEveElementList*   m_rpcEndcapElements;
   TEveElementList*   m_GEMElements;
   TEveElementList*   m_ME0Elements;

};


#endif
