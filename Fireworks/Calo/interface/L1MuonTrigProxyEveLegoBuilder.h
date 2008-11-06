#ifndef Fireworks_Calo_L1MuonTrigProxyEveLegoBuilder_h
#define Fireworks_Calo_L1MuonTrigProxyEveLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1MuonTrigProxyEveLegoBuilder
//
/**\class L1MuonTrigProxyEveLegoBuilder L1MuonTrigProxyEveLegoBuilder.h Fireworks/Calo/interface/L1MuonTrigProxyEveLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: L1MuonTrigProxyEveLegoBuilder.h,v 1.2 2008/07/07 00:46:19 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"

#include <map>

// forward declarations
class L1MuonTrigProxyEveLegoBuilder : public FW3DLegoEveElementProxyBuilder
{

   public:
      L1MuonTrigProxyEveLegoBuilder();
      virtual ~L1MuonTrigProxyEveLegoBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      REGISTER_PROXYBUILDER_METHODS();

   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      L1MuonTrigProxyEveLegoBuilder(const L1MuonTrigProxyEveLegoBuilder&); // stop default

      const L1MuonTrigProxyEveLegoBuilder& operator=(const L1MuonTrigProxyEveLegoBuilder&); // stop default

};


#endif
