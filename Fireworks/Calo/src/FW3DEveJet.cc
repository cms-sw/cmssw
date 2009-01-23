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
// $Id: FW3DEveJet.cc,v 1.3 2008/12/12 06:07:27 dmytro Exp $
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
   TEveBoxSet(iName, iTitle),
   m_color(0)
{
   SetMainColorPtr(&m_color);
   Reset(TEveBoxSet::kBT_EllipticCone, kTRUE, 64);

   //it appears that Reset resets the color as well so we need to set it back
   Color_t color = GetMainColor();
   UChar_t trans = GetMainTransparency();

   TEveVector dir, pos;
   dir.Set(iData.px()/iData.p(), iData.py()/iData.p(), iData.pz()/iData.p());

   const double z_ecal = 306; // ECAL endcap inner surface
   const double r_ecal = 126; // ECAL barrel radius
   const double transition_angle = atan(r_ecal/z_ecal);

   double length(0);
   if ( iData.theta() < transition_angle || M_PI-iData.theta() < transition_angle )
      length = z_ecal/fabs(cos(iData.theta()));
   else
      length = r_ecal/sin(iData.theta());
   dir *= length;

   pos.Set(iData.vertex().x(),iData.vertex().y(),iData.vertex().z());
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

   double theta_size = fabs(getTheta(iData.eta()+eta_size)-getTheta(iData.eta()-eta_size));

   AddEllipticCone(pos, dir, theta_size*length, phi_size*length);

   SetMainColor(color);
   SetMainTransparency(trans);
}

FW3DEveJet::~FW3DEveJet()
{
}

//
// member functions
//

void
FW3DEveJet::SetMainColor(Color_t iColor)
{
   m_color = iColor;
   TEveBoxSet::SetMainColor(iColor);
   DigitColor( iColor, GetMainTransparency() );
}

void
FW3DEveJet::SetMainTransparency(UChar_t iTrans)
{
   TEveBoxSet::SetMainTransparency(iTrans);
   DigitColor( GetMainColor(), iTrans );
}

//
// const member functions
//

//
// static member functions
//
