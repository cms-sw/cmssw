#ifndef Fireworks_Calo_L1MuonTrigProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_L1MuonTrigProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1MuonTrigProxyRhoPhiZ2DBuilder
//
/**\class L1MuonTrigProxyRhoPhiZ2DBuilder L1MuonTrigProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/L1MuonTrigProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: L1MuonTrigProxyRhoPhiZ2DBuilder.h,v 1.1 2008/06/13 18:06:35 srappocc Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"


// forward declarations

class TEveGeoShapeExtract;
class L1MuonTrigProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      L1MuonTrigProxyRhoPhiZ2DBuilder();
      virtual ~L1MuonTrigProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      REGISTER_PROXYBUILDER_METHODS();

   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

      double getTheta( double eta ) { return 2*atan(exp(-eta)); }

      L1MuonTrigProxyRhoPhiZ2DBuilder(const L1MuonTrigProxyRhoPhiZ2DBuilder&); // stop default

      const L1MuonTrigProxyRhoPhiZ2DBuilder& operator=(const L1MuonTrigProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
