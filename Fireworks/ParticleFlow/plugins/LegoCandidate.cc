#include "LegoCandidate.h"

#include "TEveCaloData.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/fwLog.h"

//______________________________________________________________________________________________________________________________________________
LegoCandidate::LegoCandidate( float eta, float phi, float energy, float et, float pt,  const FWViewContext *vc, const fireworks::Context &context )
{
    m_et = et;
    m_energy = energy;
    
    // energy auto scale
    FWViewEnergyScale *scaleE = vc->getEnergyScale( "PFenergy" );
	scaleE->setMaxVal( m_et );

    // et auto scale
    FWViewEnergyScale *scaleEt = vc->getEnergyScale( "PFet" );
	scaleEt->setMaxVal( m_et );

    float base = 0.001;     // Floor offset 1%
    
    // First vertical line
	FWViewEnergyScale *caloScale = vc->getEnergyScale("Calo");
    float val = caloScale->getPlotEt() ? m_et : m_energy;

    AddLine(eta,phi, base, 
           eta,phi, base + val*getScale(vc, context));
    
    AddMarker( 0, 1.f );
    SetMarkerStyle( 3 );
    SetMarkerSize( 0.01 );
    SetDepthTest( false );
    
    // Circle pt
    const unsigned int nLineSegments = 20;
    const double jetRadius = log( 1 + pt ) / log(10) / 5.f;

    for( unsigned int iphi = 0; iphi < nLineSegments; iphi++ )
    {
        AddLine( eta + jetRadius * cos( 2 * M_PI / nLineSegments * iphi ),
                 phi + jetRadius * sin( 2 * M_PI / nLineSegments * iphi ),
                 base,
                 eta + jetRadius * cos( 2 * M_PI / nLineSegments*( iphi+1 ) ),
                 phi + jetRadius * sin( 2 * M_PI / nLineSegments*( iphi+1 ) ),
                 base );
    }
}

//______________________________________________________________________________________________________________________________________________
float
LegoCandidate::getScale( const FWViewContext *vc, const fireworks::Context &context ) const
{
    float s = 0.f;
    
	FWViewEnergyScale *caloScale = vc->getEnergyScale("Calo");

    if( context.getCaloData()->Empty() && caloScale->getScaleMode() == FWViewEnergyScale::kAutoScale )
    {
        // Presume plotEt flag is same for "Calo" and particle flow
        if( caloScale->getPlotEt() )
        {
            s = vc->getEnergyScale( "PFet" )->getMaxVal();
        }
        else
        {
            s = vc->getEnergyScale( "PFenergy" )->getMaxVal();
        }

        // Check (if this is used in simple proxy builder then assert will be better)
        if( s == 0.f )
        {
            fwLog( fwlog::kError ) << "FWLegoEvePFCandidate max value is zero !";
            s = 1.f;
        }

        // Height of TEveCaloLego is TMath::Pi(), see FWLegoViewBase::setContext()
        return TMath::Pi()/s;
    }
    else
    {
        // Height of TEveCaloLego is TMath::Pi(), see FWLegoViewBase::setContext()
        return caloScale->getValToHeight() * TMath::Pi();
    }
}

//______________________________________________________________________________________________________________________________________________
void
LegoCandidate::updateScale( const FWViewContext *vc, const fireworks::Context &context )
{
    FWViewEnergyScale *caloScale = vc->getEnergyScale( "Calo" );
    float val = caloScale->getPlotEt() ? m_et : m_energy;

    // printf("update scale %f \n", getScale(vc, context)); fflush(stdout);

    // Resize first line
    TEveChunkManager::iterator li( GetLinePlex() );
    li.next();
    TEveStraightLineSet::Line_t &l = * (TEveStraightLineSet::Line_t*) li();
    l.fV2[2] = l.fV1[2] + val*getScale(vc, context);

    // move end point
    TEveChunkManager::iterator mi( GetMarkerPlex() );
    mi.next();
    TEveStraightLineSet::Marker_t &m = * (TEveStraightLineSet::Marker_t *) mi();
    m.fV[2] = l.fV2[2];
}
