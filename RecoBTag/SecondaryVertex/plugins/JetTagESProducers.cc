#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputerWrapper.h"

#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"
#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputerV2.h"
#include "RecoBTag/SecondaryVertex/interface/CombinedSVSoftLeptonComputer.h"
#include "RecoBTag/SecondaryVertex/interface/GhostTrackComputer.h"
#include "RecoBTag/SecondaryVertex/interface/SimpleSecondaryVertexComputer.h"

namespace { // C++ template pointer want "external" linkage, so here we go
	extern const char ipTagInfos[] = "ipTagInfos";
	extern const char svTagInfos[] = "svTagInfos";
	extern const char muonTagInfos[] = "muonTagInfos";
	extern const char elecTagInfos[] = "elecTagInfos";
}

typedef GenericMVAJetTagComputerWrapper<CombinedSVComputer,
	reco::TrackIPTagInfo,         ipTagInfos,
	reco::SecondaryVertexTagInfo, svTagInfos> CombinedSVJetTagComputer;

typedef GenericMVAJetTagComputerWrapper<CombinedSVComputerV2,
	reco::TrackIPTagInfo,         ipTagInfos,
	reco::SecondaryVertexTagInfo, svTagInfos> CombinedSVJetTagComputerV2;

typedef GenericMVAJetTagComputerWrapper<CombinedSVSoftLeptonComputer,
	reco::TrackIPTagInfo,         ipTagInfos,
	reco::SecondaryVertexTagInfo, svTagInfos,
	reco::SoftLeptonTagInfo, muonTagInfos,
	reco::SoftLeptonTagInfo, elecTagInfos> CombinedSVSoftLeptonJetTagComputer;

typedef GenericMVAJetTagComputerWrapper<GhostTrackComputer,
	reco::TrackIPTagInfo,         ipTagInfos,
	reco::SecondaryVertexTagInfo, svTagInfos> GhostTrackJetTagComputer;

typedef JetTagComputerESProducer<CombinedSVJetTagComputer> CombinedSecondaryVertexESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CombinedSecondaryVertexESProducer);

typedef JetTagComputerESProducer<CombinedSVJetTagComputerV2> CombinedSecondaryVertexESProducerV2;
DEFINE_FWK_EVENTSETUP_MODULE(CombinedSecondaryVertexESProducerV2);

typedef JetTagComputerESProducer<CombinedSVSoftLeptonJetTagComputer> CombinedSecondaryVertexSoftLeptonESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CombinedSecondaryVertexSoftLeptonESProducer);

typedef JetTagComputerESProducer<GhostTrackJetTagComputer> GhostTrackESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(GhostTrackESProducer);

typedef JetTagComputerESProducer<SimpleSecondaryVertexComputer> SimpleSecondaryVertexESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(SimpleSecondaryVertexESProducer);
