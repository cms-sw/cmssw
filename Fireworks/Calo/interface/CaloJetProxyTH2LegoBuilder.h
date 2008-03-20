#ifndef Fireworks_Calo_CaloJetProxyTH2LegoBuilder_h
#define Fireworks_Calo_CaloJetProxyTH2LegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetProxyTH2LegoBuilder
// 
/**\class CaloJetProxyTH2LegoBuilder CaloJetProxyTH2LegoBuilder.h Fireworks/Calo/interface/CaloJetProxyTH2LegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: CaloJetProxyTH2LegoBuilder.h,v 1.3 2008/03/07 09:06:47 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

// forward declarations
class TH2F;
class CaloJetProxyTH2LegoBuilder : public FW3DLegoDataProxyBuilder
{

   public:
      CaloJetProxyTH2LegoBuilder();
      virtual ~CaloJetProxyTH2LegoBuilder();

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

      CaloJetProxyTH2LegoBuilder(const CaloJetProxyTH2LegoBuilder&); // stop default

      const CaloJetProxyTH2LegoBuilder& operator=(const CaloJetProxyTH2LegoBuilder&); // stop default

      double deltaR( double, double, double, double );
      // ---------- member data --------------------------------
      const TH2F* m_product;
};


#endif
