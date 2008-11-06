// -*- C++ -*-
#ifndef Fireworks_Calo_PhotonsProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_PhotonsProxyRhoPhiZ2DBuilder_h
//
// Package:     Calo
// Class  :     PhotonsProxyRhoPhiZ2DBuilder
//
/**\class PhotonsProxyRhoPhiZ2DBuilder PhotonsProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/PhotonsProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: PhotonsProxyRhoPhiZ2DBuilder.h,v 1.1 2008/09/21 13:16:22 jmuelmen Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"

// forward declarations

class TEveGeoShapeExtract;

class PhotonsProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      PhotonsProxyRhoPhiZ2DBuilder();
      virtual ~PhotonsProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

   private:
      PhotonsProxyRhoPhiZ2DBuilder(const PhotonsProxyRhoPhiZ2DBuilder&); // stop default

      const PhotonsProxyRhoPhiZ2DBuilder& operator=(const PhotonsProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
