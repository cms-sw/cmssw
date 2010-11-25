#include "FWPFPatJetLegoProxyBuilder.h"

 /*****************************************************\
(         CONSTRUCTORS/DESTRUCTOR                       )
 \*****************************************************/

template<class T> FWPFPatJetLegoProxyBuilder<T>::FWPFPatJetLegoProxyBuilder(){}
template<class T> FWPFPatJetLegoProxyBuilder<T>::~FWPFPatJetLegoProxyBuilder(){}

 /******************************************************\
(               MEMBER FUNCTIONS                         )
 \******************************************************/

template<class T> void
FWPFPatJetLegoProxyBuilder<T>::build(const T& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* vc)
{
   std::vector<reco::PFCandidatePtr > consts = iData.getPFConstituents();

   typedef std::vector<reco::PFCandidatePtr >::const_iterator IC;

   for(IC ic=consts.begin();
       ic!=consts.end(); ++ic) {

      const reco::PFCandidatePtr pfCandPtr = *ic;

      FWLegoEvePFCandidate* evePFCandidate = new FWLegoEvePFCandidate( *pfCandPtr, vc, FWProxyBuilderBase::context() );

      evePFCandidate->SetLineWidth(3);
      evePFCandidate->SetMarkerColor(FWProxyBuilderBase::item()->defaultDisplayProperties().color());
      fireworks::setTrackTypePF( (*pfCandPtr), evePFCandidate);
      FWProxyBuilderBase::setupAddElement( evePFCandidate, &oItemHolder );
   }
}

template<class T> void
FWPFPatJetLegoProxyBuilder<T>::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   // loop items in product
   for (TEveElement::List_i i = parent->BeginChildren(); i!= parent->EndChildren(); ++i)
   {
      if ((*i)->HasChildren())
      {
         // loop elements for the reco::PFJet item
         for (TEveElement::List_i j = (*i)->BeginChildren(); j != (*i)->EndChildren(); ++j)
         {
            FWLegoEvePFCandidate* cand = dynamic_cast<FWLegoEvePFCandidate*> (*j);
            cand->updateScale( vc, FWProxyBuilderBase::context());
         }
      }
   }
}

template<class T> void
FWPFPatJetLegoProxyBuilder<T>::localModelChanges(const FWModelId& iId, TEveElement* parent, FWViewType::EType viewType, const FWViewContext* vc)
{
   // line set marker is not same color as line, have to fix it here
   if ((parent)->HasChildren())
   {
      for (TEveElement::List_i j = parent->BeginChildren(); j != parent->EndChildren(); ++j)
      {
         FWLegoEvePFCandidate* cand = dynamic_cast<FWLegoEvePFCandidate*> (*j);
         const FWDisplayProperties& dp = FWProxyBuilderBase::item()->modelInfo(iId.index()).displayProperties();
         cand->SetMarkerColor( dp.color());
         cand->ElementChanged();
      }
   }
}

class FWPatJetLegoProxyBuilder : public FWPFPatJetLegoProxyBuilder<pat::Jet> {
public:
   FWPatJetLegoProxyBuilder(){}
   virtual ~FWPatJetLegoProxyBuilder(){}

   REGISTER_PROXYBUILDER_METHODS();
};

class FWPFJetLegoProxyBuilder : public FWPFPatJetLegoProxyBuilder<reco::PFJet> {
public:
   FWPFJetLegoProxyBuilder(){}
   virtual ~FWPFJetLegoProxyBuilder(){}

   REGISTER_PROXYBUILDER_METHODS();
};

template class FWPFPatJetLegoProxyBuilder<reco::PFJet>;
template class FWPFPatJetLegoProxyBuilder<pat::Jet>;

REGISTER_FWPROXYBUILDER(FWPFJetLegoProxyBuilder, reco::PFJet, "PF Jet", FWViewType::kLegoPFECALBit);
REGISTER_FWPROXYBUILDER(FWPatJetLegoProxyBuilder, pat::Jet, "PF PatJet", FWViewType::kLegoPFECALBit);
