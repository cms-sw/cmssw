#include "FWPFClusterRPProxyBuilder.h"

//______________________________________________________________________________________________________________________________________________
float
FWPFClusterRPProxyBuilder::calculateEt( const reco::PFCluster &cluster )
{
   float et = 0.f;
   float E = cluster.energy();
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
FWPFClusterRPProxyBuilder::build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   float et, energy, phi;
   float size = 1.f;    // Stored in scale
   float ecalR = context().caloR1();
   TEveVector vec;

   et = calculateEt( iData );
   energy = iData.energy();
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

//______________________________________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFClusterRPProxyBuilder, reco::PFCluster, "PF Cluster", FWViewType::kRhoPhiPFBit );
