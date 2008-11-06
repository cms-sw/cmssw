#ifndef Fireworks_Muons_PatMuonsProxyEveLegoBuilder_h
#define Fireworks_Muons_PatMuonsProxyEveLegoBuilder_h
// -*- C++ -*-
//
// Package:     Muons
// Class  :     PatMuonsProxyEveLegoBuilder
//
/**\class PatMuonsProxyEveLegoBuilder PatMuonsProxyEveLegoBuilder.h Fireworks/PatMuons/interface/PatMuonsProxyEveLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: PatMuonsProxyEveLegoBuilder.h,v 1.1 2008/09/26 07:15:41 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"

// forward declarations
class PatMuonsProxyEveLegoBuilder : public FW3DLegoEveElementProxyBuilder
{

   public:
      PatMuonsProxyEveLegoBuilder();
      virtual ~PatMuonsProxyEveLegoBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      PatMuonsProxyEveLegoBuilder(const PatMuonsProxyEveLegoBuilder&); // stop default

      const PatMuonsProxyEveLegoBuilder& operator=(const PatMuonsProxyEveLegoBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
