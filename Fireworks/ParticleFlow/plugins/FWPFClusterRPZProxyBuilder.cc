#include "FWPFClusterRPZProxyBuilder.h"

//______________________________________________________________________________
FWPFClusterRPZProxyBuilder::FWPFClusterRPZProxyBuilder()
{
   m_clusterUtils = new FWPFClusterRPZUtils();
}

//______________________________________________________________________________
FWPFClusterRPZProxyBuilder::~FWPFClusterRPZProxyBuilder()
{
   delete m_clusterUtils;
}

//______________________________________________________________________________
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

//______________________________________________________________________________
void
FWPFClusterRPZProxyBuilder::sharedBuild( const reco::PFCluster &iData, unsigned int iIndex, 
                                         TEveElement &oItemHolder, const FWViewContext *vc, float r )
{
   /* Handles RhoPhi view */
   TEveScalableStraightLineSet *ls;
   TEveVector centre = TEveVector( iData.x(), iData.y(), iData.z() );
   const FWDisplayProperties &dp = item()->defaultDisplayProperties();
   float energy, et;
   
   energy = iData.energy();
   et = FWPFMaths::calculateEt( centre, energy );
   context().voteMaxEtAndEnergy( et, energy );

   ls = m_clusterUtils->buildRhoPhiClusterLineSet( iData, vc, energy, et, r );
   ls->SetLineColor( dp.color() );
   m_clusters.push_back( ScalableLines( ls, et, energy, vc ) );
   setupAddElement( ls, &oItemHolder );
}

//______________________________________________________________________________
void
FWPFClusterRPZProxyBuilder::build( const reco::PFCluster &iData, unsigned int iIndex, 
                                                TEveElement &oItemHolder, const FWViewContext *vc )
{
   /* Handles RhoZ view */
   float energy, et;
   float ecalR = context().caloR1();
   float ecalZ = context().caloZ1();
   const FWDisplayProperties &dp = item()->defaultDisplayProperties();
   TEveScalableStraightLineSet *ls;
   TEveVector centre = TEveVector( iData.x(), iData.y(), iData.z() );

   energy = iData.energy();
   et = FWPFMaths::calculateEt( centre, energy );
   context().voteMaxEtAndEnergy( et, energy );

   ls = m_clusterUtils->buildRhoZClusterLineSet( iData, vc, context().caloTransAngle(), energy, et, ecalR, ecalZ );
   ls->SetLineColor( dp.color() );

   m_clusters.push_back( ScalableLines( ls, et, energy, vc ) );
   setupAddElement( ls, &oItemHolder ); 
}

//______________________________________________________________________________
void
FWPFEcalClusterRPZProxyBuilder::build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   PFLayer::Layer layer = iData.layer();
   const FWEventItem::ModelInfo &info = item()->modelInfo( iIndex );
   if( info.displayProperties().isVisible() )
   {
      if( layer < 0 )
         sharedBuild( iData, iIndex, oItemHolder, vc, FWPFGeom::caloR1() );
      else
         sharedBuild( iData, iIndex, oItemHolder, vc, FWPFGeom::caloR2() );
   }
}

//______________________________________________________________________________
void
FWPFHcalClusterRPZProxyBuilder::build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   const FWEventItem::ModelInfo &info = item()->modelInfo( iIndex );
   if( info.displayProperties().isVisible() )
      sharedBuild( iData, iIndex, oItemHolder, vc, FWPFGeom::caloR1() );
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFClusterRPZProxyBuilder, reco::PFCluster, "PF Cluster", FWViewType::kRhoZBit );
REGISTER_FWPROXYBUILDER( FWPFEcalClusterRPZProxyBuilder, reco::PFCluster, "PF Cluster", FWViewType::kRhoPhiPFBit );
REGISTER_FWPROXYBUILDER( FWPFHcalClusterRPZProxyBuilder, reco::PFCluster, "PF Cluster", FWViewType::kRhoPhiBit );
