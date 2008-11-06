#ifndef Fireworks_Calo_PatJetProxyEveLegoBuilder_h
#define Fireworks_Calo_PatJetProxyEveLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     PatJetProxyEveLegoBuilder
//
/**\class PatJetProxyEveLegoBuilder PatJetProxyEveLegoBuilder.h Fireworks/Calo/interface/PatJetProxyEveLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: PatJetProxyEveLegoBuilder.h,v 1.1 2008/09/26 07:40:12 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"

// forward declarations
class PatJetProxyEveLegoBuilder : public FW3DLegoEveElementProxyBuilder
{

   public:
      PatJetProxyEveLegoBuilder();
      virtual ~PatJetProxyEveLegoBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      PatJetProxyEveLegoBuilder(const PatJetProxyEveLegoBuilder&); // stop default

      const PatJetProxyEveLegoBuilder& operator=(const PatJetProxyEveLegoBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
