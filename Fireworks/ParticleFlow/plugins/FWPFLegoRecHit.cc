#include "FWPFLegoRecHit.h"
#include "FWPFEcalRecHitLegoProxyBuilder.h"
//______________________________________________________________________________________________________
FWPFLegoRecHit::FWPFLegoRecHit( const std::vector<TEveVector> &corners, TEveElement *comp, FWPFEcalRecHitLegoProxyBuilder *pb,
                                const FWViewContext *vc, float e, float et )
   : m_builder(pb), m_energy(e), m_et(et)
{
   buildTower( corners, vc );
   buildLineSet( corners, vc );
   
   pb->setupAddElement( m_tower, comp );
   pb->setupAddElement( m_ls, comp );
}

//______________________________________________________________________________________________________
void
FWPFLegoRecHit::setupEveBox( const std::vector<TEveVector> &corners )
{
   for( size_t i = 0; i < corners.size(); ++i )
      m_tower->SetVertex( i, corners[i] );

   m_tower->SetPickable( true );
   //   m_tower->SetDrawFrame(true);
   m_tower->SetLineWidth( 1.0 );
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

   float z = caloScale->getValToHeight() * val;
   if( z < 0 )
      z *= -1;
   z += 0.001;


   TEveVector c = m_builder->calculateCentre(corners);
   float d = log( 1 + val ) /m_builder->getMaxValLog(caloScale->getPlotEt());
   d =  (corners[1].fX - corners[0].fX) * 0.5 *d;
   m_ls->AddLine(c.fX - d, c.fY -d, z, c.fX + d, c.fY -d, z);
   m_ls->AddLine(c.fX + d, c.fY -d, z, c.fX + d, c.fY +d, z);
   m_ls->AddLine(c.fX + d, c.fY +d, z, c.fX - d, c.fY +d, z);
   m_ls->AddLine(c.fX - d, c.fY +d, z, c.fX - d, c.fY -d, z);


   m_ls->SetMarkerStyle(1);
   m_ls->AddLine(c.fX, c.fY, z, c.fX, c.fY, z);
   m_ls->AddMarker(0, 0.);
}

//______________________________________________________________________________________________________
void
FWPFLegoRecHit::updateScale( const FWViewContext *vc )
{
   FWViewEnergyScale *caloScale = vc->getEnergyScale( "Calo" );
   float val = caloScale->getPlotEt() ? m_et : m_energy;
   float scale = caloScale->getValToHeight() * val;

   // printf("scale %f %f\n",  caloScale->getValToHeight(), val);

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
   while( li.next() )
   {
      TEveStraightLineSet::Line_t &l = *( TEveStraightLineSet::Line_t* ) li();
      l.fV1[2] = scale + 0.001;
      l.fV2[2] = scale + 0.001;
   }
}
