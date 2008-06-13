#ifndef Fireworks_Calo_L1EmTrigProxyEveLegoBuilder_h
#define Fireworks_Calo_L1EmTrigProxyEveLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1EmTrigProxyEveLegoBuilder
// 
/**\class L1EmTrigProxyEveLegoBuilder L1EmTrigProxyEveLegoBuilder.h Fireworks/Calo/interface/L1EmTrigProxyEveLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: L1EmTrigProxyEveLegoBuilder.h,v 1.2 2008/06/09 19:54:03 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

#include <map>

// forward declarations
class L1EmTrigProxyEveLegoBuilder : public FW3DLegoDataProxyBuilder
{

   public:
      L1EmTrigProxyEveLegoBuilder();
      virtual ~L1EmTrigProxyEveLegoBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      REGISTER_PROXYBUILDER_METHODS();
   
   private:
      virtual void build(const FWEventItem* iItem, 
			 TEveElementList** product);

      L1EmTrigProxyEveLegoBuilder(const L1EmTrigProxyEveLegoBuilder&); // stop default

      const L1EmTrigProxyEveLegoBuilder& operator=(const L1EmTrigProxyEveLegoBuilder&); // stop default

};


#endif
