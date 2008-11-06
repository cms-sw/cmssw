#ifndef Fireworks_Calo_L1MetProxyEveLegoBuilder_h
#define Fireworks_Calo_L1MetProxyEveLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1MetProxyEveLegoBuilder
//
/**\class L1MetProxyEveLegoBuilder L1MetProxyEveLegoBuilder.h Fireworks/Calo/interface/L1MetProxyEveLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: L1MetProxyEveLegoBuilder.h,v 1.1 2008/07/16 13:50:59 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"

// forward declarations
class L1MetProxyEveLegoBuilder : public FW3DLegoEveElementProxyBuilder
{

   public:
      L1MetProxyEveLegoBuilder();
      virtual ~L1MetProxyEveLegoBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      L1MetProxyEveLegoBuilder(const L1MetProxyEveLegoBuilder&); // stop default

      const L1MetProxyEveLegoBuilder& operator=(const L1MetProxyEveLegoBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
