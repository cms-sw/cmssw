#include "FWPFPatJet3DProxyBuilder.h"
#include "Fireworks/Core/interface/fwLog.h"

//______________________________________________________________________________
template<class T> FWPFPatJet3DProxyBuilder<T>::FWPFPatJet3DProxyBuilder(){}
template<class T> FWPFPatJet3DProxyBuilder<T>::~FWPFPatJet3DProxyBuilder(){}
 
//______________________________________________________________________________
template<class T> void
FWPFPatJet3DProxyBuilder<T>::build(const T& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
{
   try {
      std::vector<reco::PFCandidatePtr> consts = iData.getPFConstituents();
      typedef std::vector<reco::PFCandidatePtr>::const_iterator IC;

      for( IC ic = consts.begin();   // If consts has no constituents then the loop simply won't execute
           ic != consts.end(); ic++ )   // and so no segmentation fault should occur
      {
         const reco::PFCandidatePtr pfCandPtr = *ic;

         TEveRecTrack t;
         t.fBeta = 1;
         t.fP = TEveVector( pfCandPtr->px(), pfCandPtr->py(), pfCandPtr->pz() );
         t.fV = TEveVector( pfCandPtr->vertex().x(), pfCandPtr->vertex().y(), pfCandPtr->vertex().z() );
         t.fSign = pfCandPtr->charge();
         TEveTrack* trk = new TEveTrack(&t, FWProxyBuilderBase::context().getTrackPropagator());
         trk->MakeTrack();
         trk->SetLineWidth(3);

         fireworks::setTrackTypePF( *pfCandPtr, trk );

         FWProxyBuilderBase::setupAddElement( trk, &oItemHolder );
      }
   }
   catch (cms::Exception& iException) {
      fwLog(fwlog::kError) << "FWPFPatJet3DProxyBuilder::build() Caught exception " << iException.what() << std::endl;
   }

}

/* Classes have been created because 'concrete' types (i.e. reco::PFJet and not T) are required to register
a proxy builder. Each class must first register it's methods so that REGISTER_FWPROXYBUILDER macro knows
about them */
//_____________________________PF_______________________________________________
class FWPFJet3DProxyBuilder : public FWPFPatJet3DProxyBuilder<reco::PFJet> {
public:
   FWPFJet3DProxyBuilder(){}
   virtual ~FWPFJet3DProxyBuilder(){}

   REGISTER_PROXYBUILDER_METHODS();
};

//_____________________________PAT______________________________________________
class FWPatJet3DProxyBuilder : public FWPFPatJet3DProxyBuilder<pat::Jet> {
public:
   FWPatJet3DProxyBuilder(){}
   virtual ~FWPatJet3DProxyBuilder(){}

   REGISTER_PROXYBUILDER_METHODS();   // Register methods ready for macro
};

//______________________________________________________________________________
template class FWPFPatJet3DProxyBuilder<reco::PFJet>;
template class FWPFPatJet3DProxyBuilder<pat::Jet>;

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER(FWPFJet3DProxyBuilder, reco::PFJet, "PF Jet", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
REGISTER_FWPROXYBUILDER(FWPatJet3DProxyBuilder, pat::Jet, "PF PatJet", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
