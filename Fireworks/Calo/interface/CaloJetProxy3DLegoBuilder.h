#ifndef Fireworks_Calo_CaloJetProxy3DLegoBuilder_h
#define Fireworks_Calo_CaloJetProxy3DLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetProxy3DLegoBuilder
// 
/**\class CaloJetProxy3DLegoBuilder CaloJetProxy3DLegoBuilder.h Fireworks/Calo/interface/CaloJetProxy3DLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: CaloJetProxy3DLegoBuilder.h,v 1.2 2008/03/06 10:17:15 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

// forward declarations
class TH2F;
class CaloJetProxy3DLegoBuilder : public FW3DLegoDataProxyBuilder
{

   public:
      CaloJetProxy3DLegoBuilder();
      virtual ~CaloJetProxy3DLegoBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------
      static void build(const FWEventItem* iItem, 
			TH2* product,
			bool selectedFlag );

      // ---------- member functions ---------------------------
      virtual void message( int type, int xbin, int ybin );

   private:
      virtual void build(const FWEventItem* iItem, 
			 TH2** product);

      CaloJetProxy3DLegoBuilder(const CaloJetProxy3DLegoBuilder&); // stop default

      const CaloJetProxy3DLegoBuilder& operator=(const CaloJetProxy3DLegoBuilder&); // stop default

      double deltaR( double, double, double, double );
      // ---------- member data --------------------------------
      const TH2F* m_product;
};


#endif
