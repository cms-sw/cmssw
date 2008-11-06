#ifndef Fireworks_Calo_L1MetGlimpseProxyBuilder_h
#define Fireworks_Calo_L1MetGlimpseProxyBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1MetGlimpseProxyBuilder
//
/**\class L1MetGlimpseProxyBuilder L1MetGlimpseProxyBuilder.h Fireworks/Calo/interface/L1MetGlimpseProxyBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: L1MetGlimpseProxyBuilder.h,v 1.1 2008/07/16 13:50:59 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWGlimpseDataProxyBuilder.h"

// forward declarations
class L1MetGlimpseProxyBuilder : public FWGlimpseDataProxyBuilder
{

   public:
      L1MetGlimpseProxyBuilder();
      virtual ~L1MetGlimpseProxyBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      L1MetGlimpseProxyBuilder(const L1MetGlimpseProxyBuilder&); // stop default

      const L1MetGlimpseProxyBuilder& operator=(const L1MetGlimpseProxyBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
