#ifndef Fireworks_Calo_FW3DEveJet_h
#define Fireworks_Calo_FW3DEveJet_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FW3DEveJet
//
/**\class FW3DEveJet FW3DEveJet.h Fireworks/Calo/interface/FW3DEveJet.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Jul  4 10:22:47 EDT 2008
// $Id: FW3DEveJet.h,v 1.2 2009/01/23 21:35:39 amraktad Exp $
//

// system include files
#include "TEveJetCone.h"

// user include files

// forward declarations
namespace reco {
   class Jet;
}

class FW3DEveJet : public TEveJetCone
{

public:
   FW3DEveJet(const reco::Jet& iJet,
              const Text_t* iName, const Text_t* iTitle="");
   virtual ~FW3DEveJet();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

private:
   FW3DEveJet(const FW3DEveJet&);    // stop default

   const FW3DEveJet& operator=(const FW3DEveJet&);    // stop default

   // ---------- member data --------------------------------
   const reco::Jet* m_jet;
};


#endif
