#include "FWPFLegoCandidate.h"

#include "TEveCaloData.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/fwLog.h"

//______________________________________________________________________________________________________________________________________________
FWPFLegoCandidate::FWPFLegoCandidate( const LegoCandidateData &lc, const FWViewContext *vc, const fireworks::Context &context )
{
   float pt = lc.pt;
   float eta = lc.eta;
   float phi = lc.phi;
   m_et = lc.et;
   m_energy = lc.energy;
   float base = 0.001;     // Floor offset 1%
    
   // First vertical line
   FWViewEnergyScale *caloScale = vc->getEnergyScale();
   float val = caloScale->getPlotEt() ? m_et : m_energy;

   AddLine(eta,phi, base, 
           eta,phi, base + val*caloScale->getScaleFactorLego());
    
   AddMarker( 0, 1.f );
   SetMarkerStyle( 3 );
   SetMarkerSize( 0.01 );
   SetDepthTest( false );
    
   // Circle pt
   const unsigned int nLineSegments = 20;
   const double jetRadius = log( 1 + pt ) / log(10) / 5.f;

   for( unsigned int iphi = 0; iphi < nLineSegments; iphi++ )
   {
      AddLine( eta + jetRadius * cos( 2 * TMath::Pi() / nLineSegments * iphi ),
               phi + jetRadius * sin( TMath::TwoPi() / nLineSegments * iphi ),
               base,
               eta + jetRadius * cos( TMath::TwoPi() / nLineSegments*( iphi+1 ) ),
               phi + jetRadius * sin( TMath::TwoPi() / nLineSegments*( iphi+1 ) ),
               base );
   }
}

//______________________________________________________________________________________________________________________________________________
void
FWPFLegoCandidate::updateScale( const FWViewContext *vc, const fireworks::Context &context )
{
   FWViewEnergyScale *caloScale = vc->getEnergyScale();
   float val = caloScale->getPlotEt() ? m_et : m_energy;

   // printf("update scale %f \n", getScale(vc, context)); fflush(stdout);

   // Resize first line
   TEveChunkManager::iterator li( GetLinePlex() );
   li.next();
   TEveStraightLineSet::Line_t &l = * (TEveStraightLineSet::Line_t*) li();
   l.fV2[2] = l.fV1[2] + val*caloScale->getScaleFactorLego();

   // move end point (marker)
   TEveChunkManager::iterator mi( GetMarkerPlex() );
   mi.next();
   TEveStraightLineSet::Marker_t &m = * (TEveStraightLineSet::Marker_t *) mi();
   m.fV[2] = l.fV2[2]; // Set to new top of line
}
