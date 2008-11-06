#ifndef Fireworks_Calo_MetProxyEveLegoBuilder_h
#define Fireworks_Calo_MetProxyEveLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     MetProxyEveLegoBuilder
//
/**\class MetProxyEveLegoBuilder MetProxyEveLegoBuilder.h Fireworks/Calo/interface/MetProxyEveLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: MetProxyEveLegoBuilder.h,v 1.2 2008/07/07 00:46:19 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"

// forward declarations
class MetProxyEveLegoBuilder : public FW3DLegoEveElementProxyBuilder
{

   public:
      MetProxyEveLegoBuilder();
      virtual ~MetProxyEveLegoBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      MetProxyEveLegoBuilder(const MetProxyEveLegoBuilder&); // stop default

      const MetProxyEveLegoBuilder& operator=(const MetProxyEveLegoBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
