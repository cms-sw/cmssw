#ifndef Fireworks_Calo_CaloJetSelectedProxy3DLegoBuilder_h
#define Fireworks_Calo_CaloJetSelectedProxy3DLegoBuilder_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     CaloJetSelectedProxy3DLegoBuilder
// 
/**\class CaloJetSelectedProxy3DLegoBuilder CaloJetSelectedProxy3DLegoBuilder.h Fireworks/Calo/interface/CaloJetSelectedProxy3DLegoBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: CaloJetSelectedProxy3DLegoBuilder.h,v 1.1 2008/01/07 14:15:16 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DLegoDataProxyBuilder.h"

// forward declarations

class CaloJetSelectedProxy3DLegoBuilder : public FW3DLegoDataProxyBuilder
{

   public:
      CaloJetSelectedProxy3DLegoBuilder();
      virtual ~CaloJetSelectedProxy3DLegoBuilder();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      virtual void build(const FWEventItem* iItem, 
			 TH2** product);

      CaloJetSelectedProxy3DLegoBuilder(const CaloJetSelectedProxy3DLegoBuilder&); // stop default

      const CaloJetSelectedProxy3DLegoBuilder& operator=(const CaloJetSelectedProxy3DLegoBuilder&); // stop default

      double deltaR( double, double, double, double );
      // ---------- member data --------------------------------

};


#endif
