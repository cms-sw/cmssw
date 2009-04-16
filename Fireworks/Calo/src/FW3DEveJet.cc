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
// $Id: FW3DEveJet.cc,v 1.4 2009/01/23 21:35:40 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Calo/interface/FW3DEveJet.h"
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
FW3DEveJet::FW3DEveJet(const reco::Jet& iData,
                       const Text_t* iName, const Text_t* iTitle) :
   TEveJetCone(iName, iTitle)
{
   SetCylinder(126, 306);
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
