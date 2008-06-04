#ifndef Fireworks_Calo_L1TrigProxyRhoPhiZ2DBuilder_h
#define Fireworks_Calo_L1TrigProxyRhoPhiZ2DBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1TrigProxyRhoPhiZ2DBuilder
// 
/**\class L1TrigProxyRhoPhiZ2DBuilder L1TrigProxyRhoPhiZ2DBuilder.h Fireworks/Calo/interface/L1TrigProxyRhoPhiZ2DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: L1TrigProxyRhoPhiZ2DBuilder.h,v 1.3 2008/05/12 15:38:00 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWRPZ2DDataProxyBuilder.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"


// forward declarations

class TEveGeoShapeExtract;
class L1TrigProxyRhoPhiZ2DBuilder : public FWRPZ2DDataProxyBuilder
{

   public:
      L1TrigProxyRhoPhiZ2DBuilder();
      virtual ~L1TrigProxyRhoPhiZ2DBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   
   private:
      virtual void buildRhoPhi(const FWEventItem* iItem,
                               TEveElementList** product);
   
      virtual void buildRhoZ(const FWEventItem* iItem, 
                               TEveElementList** product);

      double getTheta( double eta ) { return 2*atan(exp(-eta)); }
   
      L1TrigProxyRhoPhiZ2DBuilder(const L1TrigProxyRhoPhiZ2DBuilder&); // stop default

      const L1TrigProxyRhoPhiZ2DBuilder& operator=(const L1TrigProxyRhoPhiZ2DBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
