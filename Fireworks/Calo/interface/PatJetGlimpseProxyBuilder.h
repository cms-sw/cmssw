#ifndef Fireworks_Calo_PatJetGlimpseProxyBuilder_h
#define Fireworks_Calo_PatJetGlimpseProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     PatJetGlimpseProxyBuilder
//
/**\class PatJetGlimpseProxyBuilder PatJetGlimpseProxyBuilder.h Fireworks/Calo/interface/PatJetGlimpseProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: PatJetGlimpseProxyBuilder.h,v 1.1 2008/09/26 07:40:12 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"

// forward declarations
class PatJetGlimpseProxyBuilder : public FWGlimpseDataProxyBuilder
{

   public:
      PatJetGlimpseProxyBuilder();
      virtual ~PatJetGlimpseProxyBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);
      double getTheta( double eta ) { return 2*atan(exp(-eta)); }

      PatJetGlimpseProxyBuilder(const PatJetGlimpseProxyBuilder&); // stop default

      const PatJetGlimpseProxyBuilder& operator=(const PatJetGlimpseProxyBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
