#ifndef Fireworks_Calo_PatMuonsGlimpseProxyBuilder_h
#define Fireworks_Calo_PatMuonsGlimpseProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     PatMuonsGlimpseProxyBuilder
//
/**\class PatMuonsGlimpseProxyBuilder PatMuonsGlimpseProxyBuilder.h Fireworks/Calo/interface/PatMuonsGlimpseProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: PatMuonsGlimpseProxyBuilder.h,v 1.1 2008/09/26 07:15:41 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"

// forward declarations
class PatMuonsGlimpseProxyBuilder : public FWGlimpseDataProxyBuilder
{

   public:
      PatMuonsGlimpseProxyBuilder();
      virtual ~PatMuonsGlimpseProxyBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      PatMuonsGlimpseProxyBuilder(const PatMuonsGlimpseProxyBuilder&); // stop default

      const PatMuonsGlimpseProxyBuilder& operator=(const PatMuonsGlimpseProxyBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
