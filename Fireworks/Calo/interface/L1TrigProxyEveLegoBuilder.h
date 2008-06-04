#ifndef Fireworks_Calo_L1TrigProxyEveLegoBuilder_h
#define Fireworks_Calo_L1TrigProxyEveLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     L1TrigProxyEveLegoBuilder
// 
/**\class L1TrigProxyEveLegoBuilder L1TrigProxyEveLegoBuilder.h Fireworks/Calo/interface/L1TrigProxyEveLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: L1TrigProxyEveLegoBuilder.h,v 1.1 2008/03/20 09:39:25 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

#include <map>

// forward declarations
class L1TrigProxyEveLegoBuilder : public FW3DLegoDataProxyBuilder
{

   public:
      L1TrigProxyEveLegoBuilder();
      virtual ~L1TrigProxyEveLegoBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem, 
			 TEveElementList** product);

      L1TrigProxyEveLegoBuilder(const L1TrigProxyEveLegoBuilder&); // stop default

      const L1TrigProxyEveLegoBuilder& operator=(const L1TrigProxyEveLegoBuilder&); // stop default

      // ---------- member data --------------------------------
      std::map<std::string,std::string>  l1TrigNamesMap_;
};


#endif
