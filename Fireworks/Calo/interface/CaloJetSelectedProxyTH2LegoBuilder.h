#ifndef Fireworks_Calo_CaloJetSelectedProxyTH2LegoBuilder_h
#define Fireworks_Calo_CaloJetSelectedProxyTH2LegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetSelectedProxyTH2LegoBuilder
// 
/**\class CaloJetSelectedProxyTH2LegoBuilder CaloJetSelectedProxyTH2LegoBuilder.h Fireworks/Calo/interface/CaloJetSelectedProxyTH2LegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: CaloJetSelectedProxyTH2LegoBuilder.h,v 1.1 2008/03/06 10:17:14 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

// forward declarations

class CaloJetSelectedProxyTH2LegoBuilder : public FW3DLegoDataProxyBuilder
{

   public:
      CaloJetSelectedProxyTH2LegoBuilder();
      virtual ~CaloJetSelectedProxyTH2LegoBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void build(const FWEventItem* iItem, 
			 TH2** product);

      CaloJetSelectedProxyTH2LegoBuilder(const CaloJetSelectedProxyTH2LegoBuilder&); // stop default

      const CaloJetSelectedProxyTH2LegoBuilder& operator=(const CaloJetSelectedProxyTH2LegoBuilder&); // stop default

      double deltaR( double, double, double, double );
      // ---------- member data --------------------------------

};


#endif
