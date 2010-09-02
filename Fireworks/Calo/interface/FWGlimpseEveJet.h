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
// $Id: FWGlimpseEveJet.h,v 1.7 2010/06/01 11:15:30 matevz Exp $
//

// system include files
#include "TEveBoxSet.h"

// user include files

// forward declarations
namespace reco {
   class Jet;
}

class FWGlimpseEveJet : public TEveBoxSet
{
public:
   FWGlimpseEveJet(const reco::Jet* iJet,
                   const Text_t* iName, const Text_t* iTitle="");
   virtual ~FWGlimpseEveJet();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setScale(float);

   //Override these so that when the system changes color it gets
   // propagated to the 'Digit'
   void SetMainColor(Color_t);
   void SetMainTransparency(Char_t);

private:
   FWGlimpseEveJet(const FWGlimpseEveJet&);    // stop default

   const FWGlimpseEveJet& operator=(const FWGlimpseEveJet&);    // stop default

   // ---------- member data --------------------------------
   const reco::Jet* m_jet;
   //NOTE: need to hold our own color since TEveBoxSet doesn't so that
   // If we later call GetMainColor we'd always get white back
   Color_t m_color;
};


#endif
