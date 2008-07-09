// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWGlimpseEveJet
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jul  4 10:23:00 EDT 2008
// $Id: FWGlimpseEveJet.cc,v 1.1 2008/07/04 23:56:28 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Calo/interface/FWGlimpseEveJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
namespace {
   double getTheta( double eta ) { return 2*atan(exp(-eta)); }
}
//
// constructors and destructor
//
FWGlimpseEveJet::FWGlimpseEveJet(const reco::CaloJet* iJet,
                                 const Text_t* iName, const Text_t* iTitle):
TEveBoxSet(iName, iTitle),
m_jet(iJet),
m_color(0)
{
   SetMainColorPtr(&m_color);
   Reset(TEveBoxSet::kBT_EllipticCone, kTRUE, 64);
   setScale(1.0);
}

// FWGlimpseEveJet::FWGlimpseEveJet(const FWGlimpseEveJet& rhs)
// {
//    // do actual copying here;
// }

FWGlimpseEveJet::~FWGlimpseEveJet()
{
}

//
// assignment operators
//
// const FWGlimpseEveJet& FWGlimpseEveJet::operator=(const FWGlimpseEveJet& rhs)
// {
//   //An exception safe implementation is
//   FWGlimpseEveJet temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWGlimpseEveJet::setScale(float iScale)
{
   //it appears that Reset resets the color as well so we need to set it back
   Color_t color = GetMainColor();
   UChar_t trans = GetMainTransparency();
   Reset();

   double height = m_jet->et()*iScale;
   TEveVector dir, pos;
   dir.Set(m_jet->px()/m_jet->p(), m_jet->py()/m_jet->p(), m_jet->pz()/m_jet->p());
   
   dir *= height;
   pos.Set(0.0,0.0,0.0);
   double eta_size = sqrt(m_jet->etaetaMoment());
   double theta_size = fabs(getTheta(m_jet->eta()+eta_size)-getTheta(m_jet->eta()-eta_size));
   double phi_size = sqrt(m_jet->phiphiMoment());
   AddEllipticCone(pos, dir, theta_size*height, phi_size*height);

   SetMainColor(color);
   SetMainTransparency(trans);
   //DigitColor( GetMainColor(), GetMainTransparency() );
}

void
FWGlimpseEveJet::SetMainColor(Color_t iColor)
{
   m_color = iColor;
   TEveBoxSet::SetMainColor(iColor);
   DigitColor( iColor, GetMainTransparency() );
}

void
FWGlimpseEveJet::SetMainTransparency(UChar_t iTrans)
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
