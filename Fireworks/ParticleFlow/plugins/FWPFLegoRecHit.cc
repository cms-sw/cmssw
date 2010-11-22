#include "FWPFLegoRecHit.h"

//______________________________________________________________________________________________________
FWPFLegoRecHit::FWPFLegoRecHit( const std::vector<TEveVector> &corners, TEveElement *comp, FWProxyBuilderBase *pb,
                                const FWViewContext *vc, const TEveVector &centre, float e, float et )
: m_itemHolder(comp), m_centre(centre), m_energy(e), m_et(et)
{
   buildTower( corners, vc );
   buildLineSet( corners, vc );
   
   pb->setupAddElement( m_tower, m_itemHolder );
   pb->setupAddElement( m_ls, m_itemHolder );
}

//______________________________________________________________________________________________________
void
FWPFLegoRecHit::setupEveBox( const std::vector<TEveVector> &corners )
{
   for( size_t i = 0; i < corners.size(); ++i )
      m_tower->SetVertex( i, corners[i] );

   m_tower->SetPickable( true );
   m_tower->SetDrawFrame(true);
   m_tower->SetLineWidth( 1.0 );
   m_tower->SetLineColor( kBlack );
}

//______________________________________________________________________________________________________
void
FWPFLegoRecHit::convertToTower( std::vector<TEveVector> &corners, float scale )
{
   for( size_t i = 0; i < 4; ++i )
      corners[i].fZ = corners[i+4].fZ + scale;
}

//______________________________________________________________________________________________________
void
FWPFLegoRecHit::buildTower( const std::vector<TEveVector> &corners, const FWViewContext *vc )
{
   m_tower = new TEveBox( "EcalRecHitTower" );
   std::vector<TEveVector> towerCorners = corners;
   FWViewEnergyScale *caloScale = vc->getEnergyScale( "Calo" );
   float val = caloScale->getPlotEt() ? m_et : m_energy;
   float scale = caloScale->getValToHeight() * val;

   if( scale < 0 )
      scale *= -1;

   convertToTower( towerCorners, scale );
   setupEveBox( towerCorners );
}

//______________________________________________________________________________________________________
void
FWPFLegoRecHit::buildLineSet( const std::vector<TEveVector> &corners, const FWViewContext *vc )
{
   m_ls = new TEveStraightLineSet( "EcalRecHitLineSet" );
   FWViewEnergyScale *caloScale = vc->getEnergyScale( "Calo" );
   float val = caloScale->getPlotEt() ? m_et : m_energy;
   float scale = caloScale->getValToHeight() * val;

   if( scale < 0 )
      scale *= -1;

   scale += 0.001;

   m_ls->AddLine( corners[0].fX, corners[0].fY, scale, corners[1].fX, corners[1].fY, scale );
   m_ls->AddLine( corners[1].fX, corners[1].fY, scale, corners[2].fX, corners[2].fY, scale );
   m_ls->AddLine( corners[2].fX, corners[2].fY, scale, corners[3].fX, corners[3].fY, scale );
   m_ls->AddLine( corners[3].fX, corners[3].fY, scale, corners[0].fX, corners[0].fY, scale );

   m_ls->SetLineWidth( 1.0 );
}

//______________________________________________________________________________________________________
void
FWPFLegoRecHit::updateScale( const FWViewContext *vc, const fireworks::Context &context, float max )
{
   TEveVector dv;
   FWViewEnergyScale *caloScale = vc->getEnergyScale( "Calo" );
   float val = caloScale->getPlotEt() ? m_et : m_energy;
   float scale = caloScale->getValToHeight() * val;

   if( scale < 0 )
      scale *= -1;

   // Reposition top points of tower
   const Float_t *data;
   for( unsigned int i = 0; i < 4; ++i )
   {
      data = m_tower->GetVertex( i );
      m_tower->SetVertex( i, data[0], data[1], 0 );
      m_tower->SetVertex( i+4,  data[0], data[1], scale);
   }
   m_tower->StampTransBBox();

   // Scale lineset
   std::vector<TEveVector> lineSetCorners(4);
   TEveChunkManager::iterator li( m_ls->GetLinePlex() ); 
   unsigned int i = 0;  

   while( li.next() )
   {
      TEveStraightLineSet::Line_t &l = *( TEveStraightLineSet::Line_t* ) li();
      data = m_tower->GetVertex( i );
      TEveVector v1 = TEveVector( data[0], data[1], data[2] );
      if( i < 3 )
         data = m_tower->GetVertex( i );
      else
         i = 0;   // Take first corner data again for the last point
      TEveVector v2 = TEveVector( data[0], data[1], data[2] );

      v1.fX -= m_centre.fX;  v1.fY -= m_centre.fY;
      v2.fX -= m_centre.fX;  v2.fY -= m_centre.fY;
      
      l.fV1[0] = m_centre.fX + ( v1.fX * ( log( 1 + val ) / log( max ) ) );
      l.fV1[1] = m_centre.fY + ( v1.fY * ( log( 1 + val ) / log( max ) ) );
      l.fV1[2] = scale + 0.001;
      l.fV2[0] = m_centre.fX + ( v2.fX * ( log( 1 + val ) / log( max ) ) );
      l.fV2[1] = m_centre.fY + ( v2.fY * ( log( 1 + val ) / log( max ) ) );
      l.fV2[2] = scale + 0.001;

      i++;
   }
}
