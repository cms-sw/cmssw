#ifndef Fireworks_Calo_L1MetProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_L1MetProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1MetProxyRhoPhiZ2DBuilder
//
/**\class L1MetProxyRhoPhiZ2DBuilder L1MetProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/L1MetProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: L1MetProxyRhoPhiZ2DBuilder.h,v 1.1 2008/07/16 13:50:59 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"

// forward declarations

class TEveGeoShapeExtract;
class L1MetProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      L1MetProxyRhoPhiZ2DBuilder();
      virtual ~L1MetProxyRhoPhiZ2DBuilder();

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

      L1MetProxyRhoPhiZ2DBuilder(const L1MetProxyRhoPhiZ2DBuilder&); // stop default

      const L1MetProxyRhoPhiZ2DBuilder& operator=(const L1MetProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
