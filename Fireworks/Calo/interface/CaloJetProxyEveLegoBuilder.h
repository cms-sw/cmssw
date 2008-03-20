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
// $Id: CaloJetProxyEveLegoBuilder.h,v 1.1.2.1 2008/03/17 02:19:58 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

// forward declarations
class CaloJetProxyEveLegoBuilder : public FW3DLegoDataProxyBuilder
{

   public:
      CaloJetProxyEveLegoBuilder();
      virtual ~CaloJetProxyEveLegoBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
   private:
      virtual void build(const FWEventItem* iItem, 
			 TEveElementList** product);

      CaloJetProxyEveLegoBuilder(const CaloJetProxyEveLegoBuilder&); // stop default

      const CaloJetProxyEveLegoBuilder& operator=(const CaloJetProxyEveLegoBuilder&); // stop default

      // ---------- member data --------------------------------
};


#endif
