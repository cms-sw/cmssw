// -*- C++ -*-
//
// Package:     Calo
// Class  :     FW3DEveJet
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jul  4 10:23:00 EDT 2008
// $Id: FW3DEveJet.cc,v 1.5 2009/04/16 17:08:32 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Calo/interface/FW3DEveJet.h"
#include "Fireworks/Core/interface/Context.h"

#include "DataFormats/JetReco/interface/Jet.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
namespace {
   double getTheta( double eta ) {
      return 2*atan(exp(-eta));
   }
}
//
// constructors and destructor
//
FW3DEveJet::FW3DEveJet(const reco::Jet& iData, const fireworks::Context& ctx):
   TEveJetCone()
{
   SetApex(TEveVector(iData.vertex().x(),iData.vertex().y(),iData.vertex().z()));

   // check availability of consituents
   reco::Jet::Constituents c = iData.getJetConstituents();
   bool haveData = true;
   for ( reco::Jet::Constituents::const_iterator itr = c.begin(); itr != c.end(); ++itr )
      if ( !itr->isAvailable() ) {
         haveData = false;
         break;
      }
   double eta_size = 0.2;
   double phi_size = 0.2;
   if ( haveData ){
      eta_size = sqrt(iData.etaetaMoment());
      phi_size = sqrt(iData.phiphiMoment());
   }

   static float offr = 5;
   static float offz = offr/tan(ctx.caloTransAngle());
   if (iData.eta() < ctx.caloMaxEta())
      SetCylinder(ctx.caloR1(false) -offr, ctx.caloZ1(false)-offz);
   else
      SetCylinder(ctx.caloR2(false)-offr, ctx.caloZ2(false)-offz);

   AddEllipticCone(iData.eta(), iData.phi(), eta_size, phi_size);
}

FW3DEveJet::~FW3DEveJet()
{
}

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
