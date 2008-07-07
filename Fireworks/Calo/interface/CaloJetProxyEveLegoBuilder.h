#ifndef Fireworks_Calo_CaloJetProxyEveLegoBuilder_h
#define Fireworks_Calo_CaloJetProxyEveLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetProxyEveLegoBuilder
// 
/**\class CaloJetProxyEveLegoBuilder CaloJetProxyEveLegoBuilder.h Fireworks/Calo/interface/CaloJetProxyEveLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: CaloJetProxyEveLegoBuilder.h,v 1.2 2008/06/09 19:54:03 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoEveElementProxyBuilder.h"

// forward declarations
class CaloJetProxyEveLegoBuilder : public FW3DLegoEveElementProxyBuilder
{

   public:
      CaloJetProxyEveLegoBuilder();
      virtual ~CaloJetProxyEveLegoBuilder();

      // ---------- const member functions ---------------------
      REGISTER_PROXYBUILDER_METHODS();
   
      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem, 
			 TEveElementList** product);

      CaloJetProxyEveLegoBuilder(const CaloJetProxyEveLegoBuilder&); // stop default

      const CaloJetProxyEveLegoBuilder& operator=(const CaloJetProxyEveLegoBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
