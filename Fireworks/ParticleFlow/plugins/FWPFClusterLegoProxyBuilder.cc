#include "FWPFClusterLegoProxyBuilder.h"
#include "Fireworks/Candidates/interface/FWLegoCandidate.h"

//______________________________________________________________________________
void
FWPFClusterLegoProxyBuilder::localModelChanges( const FWModelId &iId, TEveElement *parent, FWViewType::EType viewType, const FWViewContext *vc )
{
    // Line set marker is not the same colour as line, fixed here
    if( ( parent )->HasChildren() )
    {
        for( TEveElement::List_i j = parent->BeginChildren(); j != parent->EndChildren(); j++ )
        {
            FWLegoCandidate *cluster = dynamic_cast<FWLegoCandidate*>( *j );
            cluster->SetMarkerColor( FWProxyBuilderBase::item()->modelInfo( iId.index() ).displayProperties().color() );
            cluster->ElementChanged();
        }
    }
}

//______________________________________________________________________________
void
FWPFClusterLegoProxyBuilder::scaleProduct( TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc )
{
   for (TEveElement::List_i i = parent->BeginChildren(); i!= parent->EndChildren(); ++i)
   {
      if ((*i)->HasChildren())
      {
         TEveElement* el = (*i)->FirstChild();  // there is only one child added in this proxy builder
         FWLegoCandidate *cluster = dynamic_cast<FWLegoCandidate*>( el );
         cluster->updateScale(vc, context());
      }
   }
}

//______________________________________________________________________________
void
FWPFClusterLegoProxyBuilder::sharedBuild( const reco::PFCluster &iData, TEveElement &oItemHolder, const FWViewContext *vc )
{
   TEveVector centre = TEveVector( iData.x(), iData.y(), iData.z() );
   float energy = iData.energy();
   float et = FWPFMaths::calculateEt( centre, energy );
   float pt = et;
   float eta = iData.eta();
   float phi = iData.phi();

   context().voteMaxEtAndEnergy( et, energy );

   FWLegoCandidate *cluster = new FWLegoCandidate( vc, FWProxyBuilderBase::context(), energy, et, pt, eta, phi );
   cluster->SetMarkerColor( FWProxyBuilderBase::item()->defaultDisplayProperties().color() );
   setupAddElement( cluster, &oItemHolder );
}

//______________________________ECAL____________________________________________
void
FWPFEcalClusterLegoProxyBuilder::build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   PFLayer::Layer layer = iData.layer();
   if( layer < 0 )
      sharedBuild( iData, oItemHolder, vc ); 
}

//______________________________HCAL____________________________________________
void
FWPFHcalClusterLegoProxyBuilder::build( const reco::PFCluster &iData, unsigned int iIndex, TEveElement &oItemHolder, const FWViewContext *vc )
{
   PFLayer::Layer layer = iData.layer();
   if( layer > 0 )
      sharedBuild( iData, oItemHolder, vc );
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER( FWPFEcalClusterLegoProxyBuilder, reco::PFCluster, "PF Cluster", FWViewType::kLegoPFECALBit );
REGISTER_FWPROXYBUILDER( FWPFHcalClusterLegoProxyBuilder, reco::PFCluster, "PF Cluster", FWViewType::kLegoBit );
