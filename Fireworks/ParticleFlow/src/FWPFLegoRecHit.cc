#include "Fireworks/ParticleFlow/interface/FWPFLegoRecHit.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"

//______________________________________________________________________________
FWPFLegoRecHit::FWPFLegoRecHit( const std::vector<TEveVector> &corners, TEveElement *comp, FWProxyBuilderBase*pb,
                                const FWViewContext *vc, float e, float et )
   : m_energy(e), m_et(et), m_isTallest(false)
{
   buildTower( corners, vc );
   buildLineSet( corners, vc );

   pb->setupAddElement( m_tower, comp );
   pb->setupAddElement( m_ls, comp );
}

//______________________________________________________________________________
void
FWPFLegoRecHit::setupEveBox( std::vector<TEveVector> &corners, float scale )
{
   for( size_t i = 0; i < 4; ++i )
   {
      int j = i + 4;
      corners[i+4].fZ = corners[i].fZ + scale;
      m_tower->SetVertex( i, corners[i] );
      m_tower->SetVertex( j, corners[j] );
   }

   m_tower->SetPickable( true );
   m_tower->SetDrawFrame(false);
   m_tower->SetLineWidth( 1.0 );
   m_tower->SetLineColor( kBlack );
}

//______________________________________________________________________________
void
FWPFLegoRecHit::buildTower( const std::vector<TEveVector> &corners, const FWViewContext *vc )
{
   m_tower = new TEveBox( "EcalRecHitTower" );
   std::vector<TEveVector> towerCorners = corners;
   FWViewEnergyScale *caloScale = vc->getEnergyScale();
   float val = caloScale->getPlotEt() ? m_et : m_energy;
   float scale = caloScale->getScaleFactorLego() * val;

   if( scale < 0 )
      scale *= -1;

   setupEveBox( towerCorners, scale );
}

//______________________________________________________________________________
void
FWPFLegoRecHit::buildLineSet( const std::vector<TEveVector> &corners, const FWViewContext *vc )
{
   m_ls = new TEveStraightLineSet( "EcalRecHitLineSet" );

   // no need to set anything, all is re-set in updateScales()
   // reserve space for square outline
   TEveVector c;
   m_ls->AddLine( c.fX, c.fY, c.fZ, c.fX, c.fY, c.fZ );
   m_ls->AddLine( c.fX, c.fY, c.fZ, c.fX, c.fY, c.fZ );
   m_ls->AddLine( c.fX, c.fY, c.fZ, c.fX, c.fY, c.fZ );
   m_ls->AddLine( c.fX, c.fY, c.fZ, c.fX, c.fY, c.fZ );

   // last line is trick to add a marker in line set
   m_ls->SetMarkerStyle( 1 );
   m_ls->AddLine( c.fX, c.fY, c.fZ, c.fX, c.fY, c.fZ );
   m_ls->AddMarker( 0, 0. );
}

//______________________________________________________________________________
void
FWPFLegoRecHit::updateScale( const FWViewContext *vc, float maxLogVal )
{
   FWViewEnergyScale *caloScale = vc->getEnergyScale();
   float val = caloScale->getPlotEt() ? m_et : m_energy;
   float scale = caloScale->getScaleFactorLego() * val;

   // printf("scale %f %f\n",  caloScale->getValToHeight(), val);

   if( scale < 0 )
      scale *= -1;

   // Reposition top points of tower
   const float *data;
   TEveVector c;
   for( unsigned int i = 0; i < 4; ++i )
   {
      data = m_tower->GetVertex( i );
      c.fX += data[0];
      c.fY += data[1];
      m_tower->SetVertex( i, data[0], data[1], 0 );
      m_tower->SetVertex( i+4,  data[0], data[1], scale);
   }
   c *= 0.25;
   // Scale lineset 
   float s = log( 1 + val ) / maxLogVal;
   float d = 0.5 * ( m_tower->GetVertex(1)[0]  -m_tower->GetVertex(0)[0]);
   d *= s;
   float z =  scale * 1.001;
   setLine(0, c.fX - d, c.fY -d, z, c.fX + d, c.fY -d, z);
   setLine(1, c.fX + d, c.fY -d, z, c.fX + d, c.fY +d, z);
   setLine(2, c.fX + d, c.fY +d, z, c.fX - d, c.fY +d, z);
   setLine(3, c.fX - d, c.fY +d, z, c.fX - d, c.fY -d, z);

   if( m_isTallest )
   {
      // This is the tallest tower and hence two additional lines needs scaling
      setLine( 4, c.fX - d, c.fY - d, z, c.fX + d, c.fY + d, z );
      setLine( 5, c.fX - d, c.fY + d, z, c.fX + d, c.fY - d, z );
   }

   TEveStraightLineSet::Marker_t* m = ((TEveStraightLineSet::Marker_t*)(m_ls->GetMarkerPlex().Atom(0)));
   m->fV[0] = c.fX; m->fV[1] = c.fY; m->fV[2] = z;

   // stamp changed elements
   m_tower->StampTransBBox();
   m_ls->StampTransBBox();
}

//______________________________________________________________________________
void FWPFLegoRecHit::setLine(int idx, float x1, float y1, float z1, float x2, float y2, float z2)
{
   // AMT: this func should go in TEveStraightLineSet class

   TEveStraightLineSet::Line_t* l = ((TEveStraightLineSet::Line_t*)(m_ls->GetLinePlex().Atom(idx)));

   l->fV1[0] = x1;
   l->fV1[1] = y1;
   l->fV1[2] = z1; 

   l->fV2[0] = x2;
   l->fV2[1] = y2;
   l->fV2[2] = z2;
}

//______________________________________________________________________________
void
FWPFLegoRecHit::setIsTallest( bool b )
{
   m_isTallest = b;
   
   if( m_isTallest )
   {
      TEveVector vec;
      addLine( vec, vec );
      addLine( vec, vec );
   }
}

//______________________________________________________________________________
void
FWPFLegoRecHit::addLine( float x1, float y1, float z1, float x2, float y2, float z2 )
{
   m_ls->AddLine( x1, y1, z1, x2, y2, z2 );
}

//______________________________________________________________________________
void
FWPFLegoRecHit::addLine( const TEveVector &v1, const TEveVector &v2 )
{
   m_ls->AddLine(v1.fX, v1.fY, v1.fZ, v2.fX, v2.fY, v2.fZ);
}
