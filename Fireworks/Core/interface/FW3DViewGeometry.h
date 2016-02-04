#ifndef Fireworks_Core_FW3DViewGeometry_h
#define Fireworks_Core_FW3DViewGeometry_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DViewGeometry
// 
/**\class FW3DViewGeometry FW3DViewGeometry.h Fireworks/Core/interface/FW3DViewGeometry.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Thu Mar 25 22:06:52 CET 2010
// $Id: FW3DViewGeometry.h,v 1.8 2011/09/27 03:04:58 amraktad Exp $
//

#include "Fireworks/Core/interface/FWViewGeometryList.h"

// forward declarations

namespace fireworks
{
   class Context;
}

class FW3DViewGeometry : public FWViewGeometryList
{

public:
   FW3DViewGeometry( const fireworks::Context& context );
   virtual ~FW3DViewGeometry();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   void showMuonBarrel( bool );
   void showMuonBarrelFull( bool );
   void showMuonEndcap( bool );
   void showPixelBarrel( bool );
   void showPixelEndcap( bool );
   void showTrackerBarrel( bool );
   void showTrackerEndcap( bool );
private:
   FW3DViewGeometry(const FW3DViewGeometry&); // stop default

   const FW3DViewGeometry& operator=(const FW3DViewGeometry&); // stop default

   // ---------- member data --------------------------------

   TEveElementList*   m_muonBarrelElements;
   TEveElementList*   m_muonBarrelFullElements;
   TEveElementList*   m_muonEndcapElements;
   TEveElementList*   m_muonEndcapFullElements;
   TEveElementList*   m_pixelBarrelElements;
   TEveElementList*   m_pixelEndcapElements;
   TEveElementList*   m_trackerBarrelElements;
   TEveElementList*   m_trackerEndcapElements;
};

#endif
