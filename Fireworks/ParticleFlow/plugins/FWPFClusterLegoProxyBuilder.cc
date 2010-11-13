#include "FWPFClusterLegoProxyBuilder.h"

//______________________________________________________________________________________________________________________________________________
float
FWPFClusterLegoProxyBuilder::calculateET( const reco::PFCluster &iData )
{
    float et = 0.f;
    float E = iData.energy();
    TEveVector vec;
    
    vec.fX = iData.x();
    vec.fY = iData.y();         // Get the cluster centroid
    vec.fZ = iData.z();
    
    vec.Normalize();
    vec *= E;   
    et = vec.Perp();            // Get perpendicular vector

    return et;
}

//______________________________________________________________________________________________________________________________________________
void
FWPFClusterLegoProxyBuilder::localModelChanges( const FWModelId &iId, TEveElement *parent, FWViewType::EType viewType, const FWViewContext *vc )
{
    // Line set marker is not the same colour as line, fixed here
    if( ( parent )->HasChildren() )
    {
        for( TEveElement::List_i j = parent->BeginChildren(); j != parent->EndChildren(); j++ )
        {
            FWPFLegoCandidate *pfc = dynamic_cast<FWPFLegoCandidate*>( *j );
            const FWDisplayProperties &dp = FWProxyBuilderBase::item()->modelInfo( iId.index() ).displayProperties();
            pfc->SetMarkerColor( dp.color() );
            pfc->ElementChanged();
        }
    }
} 

//______________________________________________________________________________________________________________________________________________
void
FWPFClusterLegoProxyBuilder::build( const FWEventItem *iItem, TEveElementList *product, const FWViewContext* vc )
{
    for( int index = 0; index < static_cast<int>( iItem->size() ); index++ )
    {
        const reco::PFCluster &iData = modelData( index );
        TEveCompound *itemHolder = createCompound();
        product->AddElement( itemHolder );
        LegoCandidateData lc;

        lc.energy = iData.energy();
        lc.et = calculateET( iData );
        lc.pt = lc.et;
        lc.eta = iData.eta();
        lc.phi = iData.phi();

        FWPFLegoCandidate *cluster = new FWPFLegoCandidate( lc, vc, FWProxyBuilderBase::context() );
        cluster->SetLineWidth( 2 );
        cluster->SetMarkerColor( FWProxyBuilderBase::item()->defaultDisplayProperties().color() );
        setupAddElement( cluster, itemHolder );
    }
}

//______________________________________________________________________________________________________________________________________________
void
FWPFClusterLegoProxyBuilder::scaleProduct( TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc )
{
   for (TEveElement::List_i i = parent->BeginChildren(); i!= parent->EndChildren(); ++i)
   {
      if ((*i)->HasChildren())
      {
         TEveElement* el = (*i)->FirstChild();  // there is only one child added in this proxy builder
         FWPFLegoCandidate *cand = dynamic_cast<FWPFLegoCandidate*>( el );
         cand->updateScale(vc, context());
      }
   }
}

REGISTER_FWPROXYBUILDER( FWPFClusterLegoProxyBuilder, reco::PFCluster, "PFCluster", FWViewType::kLegoPFECALBit );
