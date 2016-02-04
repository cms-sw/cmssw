#include "FWPFClusterRPZProxyBuilder.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Base ProxyBuilder
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________________________________________________________________________
float
FWPFClusterRPZProxyBuilder::calculateEt( const reco::PFCluster &cluster, float e )
{
   float et = 0.f;
   TEveVector vec;

   vec.fX = cluster.x();
   vec.fY = cluster.y();
   vec.fZ = cluster.z();

   vec.Normalize();
   vec *= e;
   et = vec.Perp();

   return et;
}

//______________________________________________________________________________________________________________________________________________
void
FWPFClusterRPZProxyBuilder::scaleProduct( TEveElementList *parent, FWViewType::EType viewType, const FWViewContext *vc )
{
   typedef std::vector<ScalableLines> Lines_t;
   FWViewEnergyScale *caloScale = vc->getEnergyScale();

   for( Lines_t::iterator i = m_clusters.begin(); i != m_clusters.end(); ++i )
   {
      if( vc == (*i).m_vc )
      {
         float value = caloScale->getPlotEt() ? (*i).m_et : (*i).m_energy;
         (*i).m_ls->SetScale( caloScale->getScaleFactor3D() * value );
         TEveProjected *proj = *(*i).m_ls->BeginProjecteds();
         proj->UpdateProjection();
      }
   }
}

//______________________________________________________________________________________________________________________________________________
void
FWPFClusterRPZProxyBuilder::sharedBuild( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc, float R )
{
   float et, energy, phi;
   float size = 1.f;       // Stored in scale
   float ecalR = R;
   TEveVector vec;

   energy = iData.energy();
   et = calculateEt( iData, energy );
   context().voteMaxEtAndEnergy( et, energy );

   vec.fX = iData.x();
   vec.fY = iData.y();
   vec.fZ = iData.z();
   phi = vec.Phi();

   const FWDisplayProperties &dp = item()->defaultDisplayProperties();
   FWViewEnergyScale *caloScale = vc->getEnergyScale();

   TEveScalableStraightLineSet *ls = new TEveScalableStraightLineSet( "rhophiCluster" );
   ls->SetLineWidth( 4 );
   ls->SetLineColor( dp.color() );
   
   ls->SetScaleCenter( ecalR * cos( phi ), ecalR * sin( phi ), 0 );
   ls->AddLine( ecalR * cos( phi ), ecalR * sin( phi ), 0, ( ecalR + size ) * cos( phi ), ( ecalR + size ) * sin( phi ), 0 );
   ls->SetScale( caloScale->getScaleFactor3D() * ( caloScale->getPlotEt() ? et : energy ) );

   m_clusters.push_back( ScalableLines( ls, et, energy, vc ) );
   setupAddElement( ls, &oItemHolder );
}

//______________________________________________________________________________________________________________________________________________
void
FWPFClusterRPZProxyBuilder::build( const reco::PFCluster &iData, unsigned int iIndex, 
                                                TEveElement &oItemHolder, const FWViewContext *vc )
{
   const FWEventItem::ModelInfo &info = item()->modelInfo( iIndex );
   if( info.displayProperties().isVisible() )
   {
      float et, energy;
      float size = 1.f;       // Stored in scale
      float ecalR = context().caloR1();
      float ecalZ = context().caloZ1() / tan( context().caloTransAngle() );
      double theta, phi;
      double r(0);
      TEveVector vec;

      energy = iData.energy();
      et = calculateEt( iData, energy );
      context().voteMaxEtAndEnergy( et, energy );

      vec.fX = iData.x();
      vec.fY = iData.y();
      vec.fZ = iData.z();
      phi = vec.Phi();
      theta =  vec.Theta();

      const FWDisplayProperties &dp = item()->defaultDisplayProperties();
      FWViewEnergyScale *caloScale = vc->getEnergyScale();

      TEveScalableStraightLineSet *ls = new TEveScalableStraightLineSet( "rhophiCluster" );
      ls->SetLineWidth( 4 );
      ls->SetLineColor( dp.color() );

      static const float_t offr = 4;

      if ( theta < context().caloTransAngle() || TMath::Pi() - theta < context().caloTransAngle())
      {
         ecalZ = context().caloZ2() + offr / tan( context().caloTransAngle() );
         r = ecalZ / fabs( cos( theta ) );
      }
      else
      {
         r = ecalR/sin(theta);
      }

      ls->SetScaleCenter( 0., ( phi > 0 ? r * fabs( sin( theta ) ) : -r * fabs( sin( theta ) ) ), r * cos( theta ) );
      ls->AddLine( 0., ( phi > 0 ? r * fabs( sin( theta ) ) : -r * fabs( sin( theta ) ) ), r * cos( theta ),
                   0., ( phi > 0 ? ( r + size ) * fabs( sin ( theta ) ) : -( r + size ) * fabs( sin( theta) ) ), ( r + size ) * cos( theta ) );
      ls->SetScale( caloScale->getScaleFactor3D() * ( caloScale->getPlotEt() ? et : energy ) );

      m_clusters.push_back( ScalableLines( ls, et, energy, vc ) );
      setupAddElement( ls, &oItemHolder );
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// ECAL
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________________________________________________________________________
void
FWPFEcalClusterRPZProxyBuilder::build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   PFLayer::Layer layer = iData.layer();
   const FWEventItem::ModelInfo &info = item()->modelInfo( iIndex );
   if( info.displayProperties().isVisible() )
   {
      if( layer < 0 )
         sharedBuild( iData, iIndex, oItemHolder, vc, context().caloR1() );
      else
         sharedBuild( iData, iIndex, oItemHolder, vc, 177.f );
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// HCAL
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________________________________________________________________________
void
FWPFHcalClusterRPZProxyBuilder::build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   const FWEventItem::ModelInfo &info = item()->modelInfo( iIndex );
   if( info.displayProperties().isVisible() )
      sharedBuild( iData, iIndex, oItemHolder, vc, context().caloR1() );
}

//______________________________________________________________________________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFClusterRPZProxyBuilder, reco::PFCluster, "PF Cluster", FWViewType::kRhoZBit );
REGISTER_FWPROXYBUILDER( FWPFEcalClusterRPZProxyBuilder, reco::PFCluster, "PF Cluster", FWViewType::kRhoPhiPFBit );
REGISTER_FWPROXYBUILDER( FWPFHcalClusterRPZProxyBuilder, reco::PFCluster, "PF Cluster", FWViewType::kRhoPhiBit );
