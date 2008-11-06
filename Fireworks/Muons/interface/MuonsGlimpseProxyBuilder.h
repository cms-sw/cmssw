#ifndef Fireworks_Calo_MuonsGlimpseProxyBuilder_h
#define Fireworks_Calo_MuonsGlimpseProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     MuonsGlimpseProxyBuilder
//
/**\class MuonsGlimpseProxyBuilder MuonsGlimpseProxyBuilder.h Fireworks/Calo/interface/MuonsGlimpseProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: MuonsGlimpseProxyBuilder.h,v 1.1 2008/06/19 06:57:28 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"

// forward declarations
class MuonsGlimpseProxyBuilder : public FWGlimpseDataProxyBuilder
{

   public:
      MuonsGlimpseProxyBuilder();
      virtual ~MuonsGlimpseProxyBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      MuonsGlimpseProxyBuilder(const MuonsGlimpseProxyBuilder&); // stop default

      const MuonsGlimpseProxyBuilder& operator=(const MuonsGlimpseProxyBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
