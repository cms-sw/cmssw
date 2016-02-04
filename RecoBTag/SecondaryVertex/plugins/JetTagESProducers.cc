#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputerWrapper.h"

#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"
#include "RecoBTag/SecondaryVertex/interface/GhostTrackComputer.h"
#include "RecoBTag/SecondaryVertex/interface/SimpleSecondaryVertexComputer.h"

namespace { // C++ template pointer want "external" linkage, so here we go
	extern const char ipTagInfos[] = "ipTagInfos";
	extern const char svTagInfos[] = "svTagInfos";
}

typedef GenericMVAJetTagComputerWrapper<CombinedSVComputer,
	reco::TrackIPTagInfo,         ipTagInfos,
	reco::SecondaryVertexTagInfo, svTagInfos> CombinedSVJetTagComputer;

typedef GenericMVAJetTagComputerWrapper<GhostTrackComputer,
	reco::TrackIPTagInfo,         ipTagInfos,
	reco::SecondaryVertexTagInfo, svTagInfos> GhostTrackJetTagComputer;

typedef JetTagComputerESProducer<CombinedSVJetTagComputer> CombinedSecondaryVertexESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CombinedSecondaryVertexESProducer);

typedef JetTagComputerESProducer<GhostTrackJetTagComputer> GhostTrackESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(GhostTrackESProducer);

typedef JetTagComputerESProducer<SimpleSecondaryVertexComputer> SimpleSecondaryVertexESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(SimpleSecondaryVertexESProducer);
