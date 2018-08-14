#include "FWPFPatJetLegoProxyBuilder.h"
#include "Fireworks/Candidates/interface/FWLegoCandidate.h"

//______________________________________________________________________________
template<class T> FWPFPatJetLegoProxyBuilder<T>::FWPFPatJetLegoProxyBuilder(){}
template<class T> FWPFPatJetLegoProxyBuilder<T>::~FWPFPatJetLegoProxyBuilder(){}

//______________________________________________________________________________
template<class T> void
FWPFPatJetLegoProxyBuilder<T>::build( const T& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* vc )
{
   std::vector<reco::PFCandidatePtr > consts = iData.getPFConstituents();

   typedef std::vector<reco::PFCandidatePtr >::const_iterator IC;

   for( IC ic = consts.begin(); ic != consts.end(); ++ic ) 
   {
      const reco::PFCandidatePtr pfCandPtr = *ic;

      FWLegoCandidate *candidate = new FWLegoCandidate( vc, FWProxyBuilderBase::context(), pfCandPtr->energy(), pfCandPtr->et(),
                                                            pfCandPtr->pt(), pfCandPtr->eta(), pfCandPtr->phi() );
      candidate->SetMarkerColor( FWProxyBuilderBase::item()->defaultDisplayProperties().color() );
      fireworks::setTrackTypePF( (*pfCandPtr), candidate );
      FWProxyBuilderBase::setupAddElement( candidate, &oItemHolder );

      FWProxyBuilderBase::context().voteMaxEtAndEnergy( pfCandPtr->et(), pfCandPtr->energy() );
   }
}

//______________________________________________________________________________
template<class T> void
FWPFPatJetLegoProxyBuilder<T>::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   // loop items in product
   for( TEveElement::List_i i = parent->BeginChildren(); i!= parent->EndChildren(); ++i )
   {
      if ( ( *i )->HasChildren() )
      {
         // loop elements for the reco::PFJet item
         for( TEveElement::List_i j = (*i)->BeginChildren(); j != (*i)->EndChildren(); ++j )
         {
            FWLegoCandidate *cand = dynamic_cast<FWLegoCandidate*> ( *j );
            cand->updateScale( vc, FWProxyBuilderBase::context());
         }
      }
   }
}

//______________________________________________________________________________
template<class T> void
FWPFPatJetLegoProxyBuilder<T>::localModelChanges(const FWModelId& iId, TEveElement* parent, FWViewType::EType viewType, const FWViewContext* vc)
{
   // line set marker is not same color as line, have to fix it here
   if ( ( parent )->HasChildren() )
   {
      for( TEveElement::List_i j = parent->BeginChildren(); j != parent->EndChildren(); ++j )
      {
         FWLegoCandidate *cand = dynamic_cast<FWLegoCandidate*> ( *j );
         cand->SetMarkerColor( FWProxyBuilderBase::item()->modelInfo( iId.index() ).displayProperties().color() );
         cand->ElementChanged();
      }
   }
}

//____________________________PAT_______________________________________________
class FWPatJetLegoProxyBuilder : public FWPFPatJetLegoProxyBuilder<pat::Jet>
{
   public:
      FWPatJetLegoProxyBuilder(){}
      ~FWPatJetLegoProxyBuilder() override{}

      REGISTER_PROXYBUILDER_METHODS();
};

//____________________________PF________________________________________________
class FWPFJetLegoProxyBuilder : public FWPFPatJetLegoProxyBuilder<reco::PFJet>
{
   public:
      FWPFJetLegoProxyBuilder(){}
      ~FWPFJetLegoProxyBuilder() override{}

      REGISTER_PROXYBUILDER_METHODS();
};

//______________________________________________________________________________
template class FWPFPatJetLegoProxyBuilder<reco::PFJet>;
template class FWPFPatJetLegoProxyBuilder<pat::Jet>;

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER(FWPFJetLegoProxyBuilder, reco::PFJet, "PF Jet", FWViewType::kLegoPFECALBit | FWViewType::kLegoBit );
REGISTER_FWPROXYBUILDER(FWPatJetLegoProxyBuilder, pat::Jet, "PF PatJet", FWViewType::kLegoPFECALBit | FWViewType::kLegoBit );
