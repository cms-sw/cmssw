#ifndef Fireworks_Calo_L1EtMissTrigProxyEveLegoBuilder_h
#define Fireworks_Calo_L1EtMissTrigProxyEveLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1EtMissTrigProxyEveLegoBuilder
// 
/**\class L1EtMissTrigProxyEveLegoBuilder L1EtMissTrigProxyEveLegoBuilder.h Fireworks/Calo/interface/L1EtMissTrigProxyEveLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: L1EtMissTrigProxyEveLegoBuilder.h,v 1.2 2008/06/09 19:54:03 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

#include <map>

// forward declarations
class L1EtMissTrigProxyEveLegoBuilder : public FW3DLegoDataProxyBuilder
{

   public:
      L1EtMissTrigProxyEveLegoBuilder();
      virtual ~L1EtMissTrigProxyEveLegoBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      REGISTER_PROXYBUILDER_METHODS();
   
   private:
      virtual void build(const FWEventItem* iItem, 
			 TEveElementList** product);

      L1EtMissTrigProxyEveLegoBuilder(const L1EtMissTrigProxyEveLegoBuilder&); // stop default

      const L1EtMissTrigProxyEveLegoBuilder& operator=(const L1EtMissTrigProxyEveLegoBuilder&); // stop default

};


#endif
