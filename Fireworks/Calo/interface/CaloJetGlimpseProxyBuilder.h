#ifndef Fireworks_Calo_CaloJetGlimpseProxyBuilder_h
#define Fireworks_Calo_CaloJetGlimpseProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetGlimpseProxyBuilder
//
/**\class CaloJetGlimpseProxyBuilder CaloJetGlimpseProxyBuilder.h Fireworks/Calo/interface/CaloJetGlimpseProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: CaloJetGlimpseProxyBuilder.h,v 1.2 2008/06/26 00:27:16 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"

// forward declarations
class CaloJetGlimpseProxyBuilder : public FWGlimpseDataProxyBuilder
{

   public:
      CaloJetGlimpseProxyBuilder();
      virtual ~CaloJetGlimpseProxyBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);
      double getTheta( double eta ) { return 2*atan(exp(-eta)); }

      CaloJetGlimpseProxyBuilder(const CaloJetGlimpseProxyBuilder&); // stop default

      const CaloJetGlimpseProxyBuilder& operator=(const CaloJetGlimpseProxyBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
