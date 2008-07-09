#ifndef Fireworks_Calo_FWGlimpseEveJet_h
#define Fireworks_Calo_FWGlimpseEveJet_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWGlimpseEveJet
// 
/**\class FWGlimpseEveJet FWGlimpseEveJet.h Fireworks/Calo/interface/FWGlimpseEveJet.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jul  4 10:22:47 EDT 2008
// $Id: FWGlimpseEveJet.h,v 1.1 2008/07/04 23:56:27 chrjones Exp $
//

// system include files
#include "TEveBoxSet.h"

// user include files
#include "Fireworks/Core/interface/FWEveValueScaled.h"

// forward declarations
namespace reco {
   class CaloJet;
}

class FWGlimpseEveJet : public TEveBoxSet, public FWEveValueScaled
{

   public:
      FWGlimpseEveJet(const reco::CaloJet* iJet,
                      const Text_t* iName, const Text_t* iTitle="");
      virtual ~FWGlimpseEveJet();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setScale(float);
   
      //Override these so that when the system changes color it gets
      // propagated to the 'Digit'
      void SetMainColor(Color_t);
      void SetMainTransparency(UChar_t);

   private:
      FWGlimpseEveJet(const FWGlimpseEveJet&); // stop default

      const FWGlimpseEveJet& operator=(const FWGlimpseEveJet&); // stop default

      // ---------- member data --------------------------------
      const reco::CaloJet* m_jet;
      Color_t m_color;
};


#endif
