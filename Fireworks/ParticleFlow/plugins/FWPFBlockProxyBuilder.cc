#include "FWPFBlockProxyBuilder.h"

//______________________________________________________________________________
float
FWPFBlockProxyBuilder::calculateEt( const TEveVector &centre, float e )
{
   TEveVector vec = centre;
   float et;

   vec.Normalize();
   vec *= e;
   et = vec.Perp();

   return et;
}

//______________________________________________________________________________
void
FWPFBlockProxyBuilder::scaleProduct( TEveElementList *parent, FWViewType::EType viewType, const FWViewContext *vc )
{
   typedef std::vector<ScalableLines> Lines_t;
   FWViewEnergyScale *caloScale = vc->getEnergyScale();

   if( viewType == FWViewType::kRhoPhiPF || viewType == FWViewType::kRhoZ )
   {  /* Handle the rhophi and rhoz cluster scaling */
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
}

//______________________________________________________________________________
void
FWPFBlockProxyBuilder::setupTrackElement( const reco::PFBlockElement &blockElement, 
                                              TEveElement &oItemHolder, const FWViewContext *vc, FWViewType::EType viewType )
{
   if( blockElement.trackType( reco::PFBlockElement::DEFAULT ) )
   {
      reco::Track track = *blockElement.trackRef();
      FWPFTrackUtils *trackUtils = new FWPFTrackUtils();

      if( viewType == FWViewType::kLego || viewType == FWViewType::kLegoPFECAL )       // Lego views
      {
         TEveStraightLineSet *legoTrack = trackUtils->setupLegoTrack( track );
         legoTrack->SetRnrMarkers( true );
         setupAddElement( legoTrack, &oItemHolder );
      }
      else if( viewType == FWViewType::kRhoPhiPF || viewType == FWViewType::kRhoZ )   // Projected views
      {
         TEveTrack *trk = trackUtils->setupRPZTrack( track );
         TEvePointSet *ps = trackUtils->getCollisionMarkers( trk );
         setupAddElement( trk, &oItemHolder );
         if( ps->GetN() != 0 )
            setupAddElement( ps, &oItemHolder );
         else
            delete ps;
      }
   }
}

//______________________________________________________________________________
void
FWPFBlockProxyBuilder::setupClusterElement( const reco::PFBlockElement &blockElement, TEveElement &oItemHolder, 
                                                const FWViewContext *vc, FWViewType::EType viewType, float r )
{
   // Get reference to PFCluster
   reco::PFCluster cluster = *blockElement.clusterRef();
   TEveVector centre = TEveVector( cluster.x(), cluster.y(), cluster.z() );
   float energy = cluster.energy();
   float et = calculateEt( centre, energy );
   float pt = et;
   float eta = cluster.eta();
   float phi = cluster.phi();

   FWProxyBuilderBase::context().voteMaxEtAndEnergy( et, energy );

   if( viewType == FWViewType::kLego || viewType == FWViewType::kLegoPFECAL )
   {
      FWPFLegoCandidate *legoCluster = new FWPFLegoCandidate( vc, FWProxyBuilderBase::context(), energy, et, pt, eta, phi );
      legoCluster->SetMarkerColor( FWProxyBuilderBase::item()->defaultDisplayProperties().color() );
      setupAddElement( legoCluster, &oItemHolder );
   }
   if( viewType == FWViewType::kRhoPhiPF )
   {
      const FWDisplayProperties &dp = item()->defaultDisplayProperties();
      FWPFClusterRPZUtils *clusterUtils = new FWPFClusterRPZUtils();
      TEveScalableStraightLineSet *rpCluster = clusterUtils->buildRhoPhiClusterLineSet( cluster, vc, energy, et, r );
      rpCluster->SetLineColor( dp.color() );
      m_clusters.push_back( ScalableLines( rpCluster, et, energy, vc ) );
      setupAddElement( rpCluster, &oItemHolder );
      delete clusterUtils;
   }
   else if( viewType == FWViewType::kRhoZ )
   {
      const FWDisplayProperties &dp = item()->defaultDisplayProperties();
      FWPFClusterRPZUtils *clusterUtils = new FWPFClusterRPZUtils();
      TEveScalableStraightLineSet *rzCluster = clusterUtils->buildRhoZClusterLineSet( cluster, vc, context().caloTransAngle(),
                                                                                      energy, et, r, context().caloZ1() );
      rzCluster->SetLineColor( dp.color() );
      m_clusters.push_back( ScalableLines( rzCluster, et, energy, vc ) );
      setupAddElement( rzCluster, &oItemHolder );
      delete clusterUtils;
   }
}

//______________________________________________________________________________
void
FWPFBlockProxyBuilder::buildViewType( const reco::PFBlock &iData, unsigned int iIndex, TEveElement &oItemHolder, 
                                  FWViewType::EType viewType, const FWViewContext *vc )
{
   const edm::OwnVector<reco::PFBlockElement> &elements = iData.elements();
   
   for( unsigned int i = 0; i < elements.size(); ++i )
   {
      reco::PFBlockElement::Type type = elements[i].type();
      switch( type )
      {
         case 1:  // TRACK
            if( e_builderType == BASE )
               setupTrackElement( elements[i], oItemHolder, vc, viewType );
         break;

         case 4:  // ECAL
            if( e_builderType == ECAL )
               setupClusterElement( elements[i], oItemHolder, vc, viewType, context().caloR1() );
         break;

         case 5:  // HCAL
            if( e_builderType == HCAL )
            {
               if( viewType == FWViewType::kRhoPhiPF )
                  setupClusterElement( elements[i], oItemHolder, vc, viewType, 177.7 );
               else  // RhoZ
                  setupClusterElement( elements[i], oItemHolder, vc, viewType, context().caloR1() );
            }
         break;

         default: // Ignore anything that isn't wanted
         break;
      }
   }
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFBlockProxyBuilder, reco::PFBlock, "PF Block", FWViewType::kRhoPhiPFBit | FWViewType::kLegoBit |
                                                                               FWViewType::kRhoZBit | FWViewType::kLegoPFECALBit );
REGISTER_FWPROXYBUILDER( FWPFBlockEcalProxyBuilder, reco::PFBlock, "PF Block", FWViewType::kLegoPFECALBit | FWViewType::kRhoPhiPFBit |
                                                                               FWViewType::kRhoZBit );
REGISTER_FWPROXYBUILDER( FWPFBlockHcalProxyBuilder, reco::PFBlock, "PF Block", FWViewType::kLegoBit | FWViewType::kRhoPhiPFBit |
                                                                               FWViewType::kRhoZBit );
