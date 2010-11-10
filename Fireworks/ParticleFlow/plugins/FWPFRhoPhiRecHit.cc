#include "TEveChunkManager.h"

#include "FWPFRhoPhiRecHit.h"

//______________________________________________________________________________________________________
FWPFRhoPhiRecHit::FWPFRhoPhiRecHit( FWProxyBuilderBase *pb, TEveCompound *iH, const FWViewContext *vc, const TEveVector &centre, 
                           float E, double lPhi, double rPhi, bool build )
: m_currentScale(1.0), m_lPhi(lPhi), m_rPhi(rPhi), m_energy(E), m_vc(vc), m_centre(centre)
{
   CalculateEt();
   if( build )   // Build immediately
      BuildRecHit( pb, iH );
}

//______________________________________________________________________________________________________
FWPFRhoPhiRecHit::~FWPFRhoPhiRecHit(){}

//______________________________________________________________________________________________________
void
FWPFRhoPhiRecHit::CalculateEt()
{
   TEveVector vec = m_centre;

   vec.Normalize();
   vec *= m_energy;
   m_et = vec.Perp();
}

//______________________________________________________________________________________________________
void
FWPFRhoPhiRecHit::PushCreationPoint( TEveVector vec )
{
   m_creationPoint.push_back( vec );
}

//______________________________________________________________________________________________________
void
FWPFRhoPhiRecHit::Add( FWProxyBuilderBase *pb, TEveCompound *itemHolder, const FWViewContext *vc, float E )
{
   FWPFRhoPhiRecHit *rh = new FWPFRhoPhiRecHit( pb, itemHolder, vc, m_centre, E, m_lPhi, m_rPhi );
   rh->PushCreationPoint( m_creationPoint[0] );
   rh->PushCreationPoint( m_creationPoint[1] );
   rh->BuildRecHit( pb, itemHolder );
   m_creationPoint[0] = rh->GetCreationPoint( 0 );
   m_creationPoint[1] = rh->GetCreationPoint( 1 );

   m_children.push_back( rh );
}

//______________________________________________________________________________________________________
void
FWPFRhoPhiRecHit::updateScale( TEveScalableStraightLineSet *ls, Double_t scale, unsigned int i )
{
   // First deal with Base
   ModScale();

   // Now the children
   for( unsigned int i = 0; i < m_children.size(); ++i )
   {
      m_children[i]->SetCorners( 0, m_creationPoint[0] );   // Set childs starting point to current top of the stack of towers
      m_children[i]->SetCorners( 1, m_creationPoint[1] );
      m_children[i]->ModScale();
      m_creationPoint[0] = m_children[i]->GetCreationPoint( 0 );   // New starting point for stacked towers
      m_creationPoint[1] = m_children[i]->GetCreationPoint( 1 );
   }
}

//______________________________________________________________________________________________________
void
FWPFRhoPhiRecHit::ModScale()
{
   FWViewEnergyScale *caloScale = m_vc->getEnergyScale( "Calo" );
   float value = caloScale->getPlotEt() ? m_et : m_energy;
   Double_t scale = caloScale->getValToHeight() * value;
   
   int a = 0;

   // Scale centres
   TEveVector sc1 = m_corners[1];   // Bottom right corner
   TEveVector sc2 = m_corners[0];   // Bottom left corner
   TEveVector v1 = sc1;         // Used to store normalized vectors
   TEveVector v2 = sc2;

   v1.Normalize();
   v2.Normalize();

   v1 *= scale;               // Now at new height
   v2 *= scale;

   // Get line parameters and scale coordinates
   TEveChunkManager::iterator li( m_ls->GetLinePlex() );
   while( li.next() )
   {
      TEveStraightLineSet::Line_t &l = *( TEveStraightLineSet::Line_t* ) li();
      switch( a )
      {
         case 0:
            // Left side of tower first
            l.fV1[0] = sc2.fX;
            l.fV1[1] = sc2.fY;
            l.fV2[0] = sc2.fX + v2.fX;
            l.fV2[1] = sc2.fY + v2.fY;
         break;

         case 1:
            // Top of tower
            l.fV1[0] = sc2.fX + v2.fX;
            l.fV1[1] = sc2.fY + v2.fY;
            l.fV2[0] = sc1.fX + v1.fX;
            l.fV2[1] = sc1.fY + v1.fY;
         break;

         case 2:
            // Right hand side of tower
            l.fV1[0] = sc1.fX + v1.fX;
            l.fV1[1] = sc1.fY + v1.fY;
            l.fV2[0] = sc1.fX;
            l.fV2[1] = sc1.fY;
         break;

         case 3:
            // Bottom of tower
            l.fV1[0] = sc1.fX;
            l.fV1[1] = sc1.fY;
            l.fV2[0] = sc2.fX;
            l.fV2[1] = sc2.fY;
         break;
      }
      a++;
   }
   TEveProjected *proj = *(m_ls)->BeginProjecteds();
   proj->UpdateProjection();
   m_currentScale = scale;

   m_creationPoint[0] = sc2 + v2;      // New top left of tower
   m_creationPoint[1] = sc1 + v1;      // New top right of tower
}

//______________________________________________________________________________________________________
void
FWPFRhoPhiRecHit::BuildRecHit( FWProxyBuilderBase *pb, TEveCompound *itemHolder )
{
   float ecalR = 129;
   float scale = 0;
   float value = 0;
   TEveVector v1, v2, v3, v4;
   TEveVector vec;

   FWViewEnergyScale *caloScale = m_vc->getEnergyScale( "Calo" );
   value = caloScale->getPlotEt() ? m_et : m_energy;
   scale = caloScale->getValToHeight() * value;

   if( m_creationPoint.size() == 0 )   // Else we already have creation point data
   {   // Base tower only
      TEveVector creationPoint1 = TEveVector( ecalR * cos( m_rPhi ), ecalR * sin( m_rPhi ), 0 );
      TEveVector creationPoint2 = TEveVector( ecalR * cos( m_lPhi ), ecalR * sin( m_lPhi ), 0 );
      m_creationPoint.push_back( creationPoint1 );
      m_creationPoint.push_back( creationPoint2 );
   }
   
   v1 = m_creationPoint[0];
   v2 = m_creationPoint[1];

   v3 = v1;
   vec = v3;
   vec.Normalize();
   v3 = v3 + ( vec * scale );
   
   v4 = v2;
   vec = v4;
   vec.Normalize();
   v4 = v4 + ( vec * scale );

   m_corners.push_back( v1 );   // Bottom left
   m_corners.push_back( v2 );   // Bottom right
   m_corners.push_back( v3 );   // Top left
   m_corners.push_back( v4 );   // Top right

   m_ls = new TEveScalableStraightLineSet( "rhophiRecHit" );
   m_ls->AddLine( m_corners[0].fX, m_corners[0].fY, 0,
               m_corners[2].fX, m_corners[2].fY, 0 );
   m_ls->AddLine( m_corners[2].fX, m_corners[2].fY, 0,
               m_corners[3].fX, m_corners[3].fY, 0 );
   m_ls->AddLine( m_corners[3].fX, m_corners[3].fY, 0,
               m_corners[1].fX, m_corners[1].fY, 0 );
   m_ls->AddLine( m_corners[1].fX, m_corners[1].fY, 0,
               m_corners[0].fX, m_corners[0].fY, 0 );

   m_creationPoint[0] = v3;
   m_creationPoint[1] = v4;
              
   pb->setupAddElement( m_ls, itemHolder );
}
