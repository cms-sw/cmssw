#include "FWPFPatJetRPZProxyBuilder.h"

 /*************************************************************************\
(			CONSTRUCTORS/DESTRUCTOR	        		    )
 \*************************************************************************/

template<class T> FWPFPatJetRPZProxyBuilder<T>::FWPFPatJetRPZProxyBuilder(){}
template<class T> FWPFPatJetRPZProxyBuilder<T>::~FWPFPatJetRPZProxyBuilder(){}

 /*************************************************************************\
(			      MEMBER FUNCTIONS	        		    )
 \*************************************************************************/

template<class T> void
FWPFPatJetRPZProxyBuilder<T>::build(const T& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
{
        std::vector<reco::PFCandidatePtr> consts = iData.getPFConstituents();

	typedef std::vector<reco::PFCandidatePtr>::const_iterator IC;

	for( IC ic = consts.begin();
	     ic != consts.end(); ic++ )
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

class FWPatJetRPZProxyBuilder : public FWPFPatJetRPZProxyBuilder<pat::Jet> {
public:
	FWPatJetRPZProxyBuilder(){}
	virtual ~FWPatJetRPZProxyBuilder(){}

	REGISTER_PROXYBUILDER_METHODS();
};

class FWPFJetRPZProxyBuilder : public FWPFPatJetRPZProxyBuilder<reco::PFJet> {
public:
	FWPFJetRPZProxyBuilder(){}
	virtual ~FWPFJetRPZProxyBuilder(){}

	REGISTER_PROXYBUILDER_METHODS();
};

template class FWPFPatJetRPZProxyBuilder<reco::PFJet>;
template class FWPFPatJetRPZProxyBuilder<pat::Jet>;

REGISTER_FWPROXYBUILDER(FWPFJetRPZProxyBuilder, reco::PFJet, "PFJet", FWViewType::kAllRPZBits);
REGISTER_FWPROXYBUILDER(FWPatJetRPZProxyBuilder, pat::Jet, "PFPatJet", FWViewType::kAllRPZBits);
