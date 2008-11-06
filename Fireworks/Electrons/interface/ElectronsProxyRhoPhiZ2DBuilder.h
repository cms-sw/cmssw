#ifndef Fireworks_Calo_ElectronsProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_ElectronsProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     ElectronsProxyRhoPhiZ2DBuilder
//
/**\class ElectronsProxyRhoPhiZ2DBuilder ElectronsProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/ElectronsProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: ElectronsProxyRhoPhiZ2DBuilder.h,v 1.3 2008/09/26 07:42:08 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"

// forward declarations

class TEveGeoShapeExtract;
namespace reco {
   class GsfElectron;
}
namespace fw {
   class NamedCounter;
}

class ElectronsProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      ElectronsProxyRhoPhiZ2DBuilder();
      virtual ~ElectronsProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
      static void buildElectronRhoPhi(const FWEventItem* iItem,
				      const reco::GsfElectron* electron,
				      TEveElementList* tList,
				      const fw::NamedCounter& counter);

      static void buildElectronRhoZ(  const FWEventItem* iItem,
				      const reco::GsfElectron* electron,
				      TEveElementList* tList,
				      const fw::NamedCounter& counter);

      // ---------- member functions ---------------------------

      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);

      virtual void buildRhoZ(const FWEventItem* iItem,
                               TEveElementList** product);

   private:
      ElectronsProxyRhoPhiZ2DBuilder(const ElectronsProxyRhoPhiZ2DBuilder&); // stop default

      const ElectronsProxyRhoPhiZ2DBuilder& operator=(const ElectronsProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
