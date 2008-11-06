#ifndef Fireworks_Calo_MetProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_MetProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     MetProxyRhoPhiZ2DBuilder
//
/**\class MetProxyRhoPhiZ2DBuilder MetProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/MetProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: MetProxyRhoPhiZ2DBuilder.h,v 1.1 2008/06/24 07:42:16 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"

// forward declarations

class TEveGeoShapeExtract;
class MetProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      MetProxyRhoPhiZ2DBuilder();
      virtual ~MetProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

      double getTheta( double eta ) { return 2*atan(exp(-eta)); }

      MetProxyRhoPhiZ2DBuilder(const MetProxyRhoPhiZ2DBuilder&); // stop default

      const MetProxyRhoPhiZ2DBuilder& operator=(const MetProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
