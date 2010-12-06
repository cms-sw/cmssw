#include "FWPFClusterRPProxyBuilder.h"

//______________________________________________________________________________________________________________________________________________
float
FWPFClusterRPProxyBuilder::calculateEt( const reco::PFCluster &cluster, float E )
{
   float et = 0.f;
   TEveVector vec;

   vec.fX = cluster.x();
   vec.fY = cluster.y();
   vec.fZ = cluster.z();

   vec.Normalize();
   vec *= E;
   et = vec.Perp();

   return et;
}

//______________________________________________________________________________________________________________________________________________
void
FWPFClusterRPProxyBuilder::sharedBuild( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc, float R )
{
   float et, energy, phi;
   float size = 1.f;    // Stored in scale
   float ecalR = R;
   TEveVector vec;

   energy = iData.energy();
   et = calculateEt( iData, energy );
   context().voteMaxEtAndEnergy(et, energy);
   
   vec.fX = iData.x();
   vec.fY = iData.y();
   vec.fZ = iData.z();
   phi = vec.Phi();

   const FWDisplayProperties &dp = item()->defaultDisplayProperties();
   FWViewEnergyScale *caloScale = vc->getEnergyScale();

   TEveScalableStraightLineSet *ls = new TEveScalableStraightLineSet( "rhophiPFCluster" );
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
FWPFClusterRPProxyBuilder::scaleProduct( TEveElementList *parent, FWViewType::EType type, const FWViewContext *vc )
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// ECAL
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________________________________________________________________________
void
FWPFEcalClusterRPProxyBuilder::build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
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

//______________________________________________________________________________________________________________________________________________
bool
FWPFEcalClusterRPProxyBuilder::visibilityModelChanges( const FWModelId &iId, TEveElement *itemHolder,
                                                   FWViewType::EType viewType, const FWViewContext *vc )
{
   const FWEventItem::ModelInfo &info = iId.item()->modelInfo( iId.index() );
   const reco::PFCluster &iData = modelData( iId.index() );
   PFLayer::Layer layer = iData.layer();

   //build
   if( info.displayProperties().isVisible() && itemHolder->NumChildren() == 0 )
   {
      if( layer < 0 )
         sharedBuild( iData, iId.index(), *itemHolder, vc, context().caloR1() );
      else
         sharedBuild( iData, iId.index(), *itemHolder, vc, 177.f );
      return true;
   }
   return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// HCAL
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________________________________________________________________________
void
FWPFHcalClusterRPProxyBuilder::build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   //PFLayer::Layer layer = iData.layer();
   const FWEventItem::ModelInfo &info = item()->modelInfo( iIndex );
   if( info.displayProperties().isVisible() )
   {
      sharedBuild( iData, iIndex, oItemHolder, vc, context().caloR1() );
   }
}

//______________________________________________________________________________________________________________________________________________
bool
FWPFHcalClusterRPProxyBuilder::visibilityModelChanges( const FWModelId &iId, TEveElement *itemHolder,
                                                   FWViewType::EType viewType, const FWViewContext *vc )
{
   const FWEventItem::ModelInfo &info = iId.item()->modelInfo( iId.index() );
   const reco::PFCluster &iData = modelData( iId.index() );

   //build
   if( info.displayProperties().isVisible() && itemHolder->NumChildren() == 0 )
   {
      sharedBuild( iData, iId.index(), *itemHolder, vc, context().caloR1() );
      return true;
   }
   return false;
}

//______________________________________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFEcalClusterRPProxyBuilder, reco::PFCluster, "PF Cluster", FWViewType::kRhoPhiPFBit );
REGISTER_FWPROXYBUILDER( FWPFHcalClusterRPProxyBuilder, reco::PFCluster, "PF Cluster", FWViewType::kRhoPhiBit );
