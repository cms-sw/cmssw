#ifndef Fireworks_Calo_L1JetTrigProxyEveLegoBuilder_h
#define Fireworks_Calo_L1JetTrigProxyEveLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1JetTrigProxyEveLegoBuilder
//
/**\class L1JetTrigProxyEveLegoBuilder L1JetTrigProxyEveLegoBuilder.h Fireworks/Calo/interface/L1JetTrigProxyEveLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: L1JetTrigProxyEveLegoBuilder.h,v 1.2 2008/07/07 00:46:19 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"

#include <map>

// forward declarations
class L1JetTrigProxyEveLegoBuilder : public FW3DLegoEveElementProxyBuilder
{

   public:
      L1JetTrigProxyEveLegoBuilder();
      virtual ~L1JetTrigProxyEveLegoBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      REGISTER_PROXYBUILDER_METHODS();

   private:
      virtual void build(const FWEventItem* iItem,
			 TEveElementList** product);

      L1JetTrigProxyEveLegoBuilder(const L1JetTrigProxyEveLegoBuilder&); // stop default

      const L1JetTrigProxyEveLegoBuilder& operator=(const L1JetTrigProxyEveLegoBuilder&); // stop default

};


#endif
