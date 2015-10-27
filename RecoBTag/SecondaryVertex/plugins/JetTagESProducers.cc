#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputerESProducer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAJetTagComputerWrapper.h"

#include "RecoBTag/SecondaryVertex/interface/CandidateBoostedDoubleSecondaryVertexComputer.h"
#include "RecoBTag/SecondaryVertex/interface/CombinedSVComputer.h"
#include "RecoBTag/SecondaryVertex/interface/CombinedSVSoftLeptonComputer.h"
#include "RecoBTag/SecondaryVertex/interface/GhostTrackComputer.h"
#include "RecoBTag/SecondaryVertex/interface/CandidateSimpleSecondaryVertexComputer.h"
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

typedef GenericMVAJetTagComputerWrapper<CombinedSVComputer,
	reco::CandIPTagInfo,         ipTagInfos,
	reco::CandSecondaryVertexTagInfo, svTagInfos> CandidateCombinedSVJetTagComputer;

//this one is actually not fully non-candidate based anymore (no backward compatibility with old SL taginfos for the moment)
typedef GenericMVAJetTagComputerWrapper<CombinedSVSoftLeptonComputer,
	reco::TrackIPTagInfo,         ipTagInfos,
	reco::SecondaryVertexTagInfo, svTagInfos,
	reco::CandSoftLeptonTagInfo, muonTagInfos,
	reco::CandSoftLeptonTagInfo, elecTagInfos> CombinedSVSoftLeptonJetTagComputer;
		
typedef GenericMVAJetTagComputerWrapper<CombinedSVSoftLeptonComputer,
	reco::CandIPTagInfo,         ipTagInfos,
	reco::CandSecondaryVertexTagInfo, svTagInfos,
	reco::CandSoftLeptonTagInfo, muonTagInfos,
	reco::CandSoftLeptonTagInfo, elecTagInfos> CandidateCombinedSVSoftLeptonJetTagComputer;

typedef GenericMVAJetTagComputerWrapper<GhostTrackComputer,
	reco::TrackIPTagInfo,         ipTagInfos,
	reco::SecondaryVertexTagInfo, svTagInfos> GhostTrackJetTagComputer;

typedef GenericMVAJetTagComputerWrapper<CombinedSVSoftLeptonComputer,
	reco::CandIPTagInfo,         ipTagInfos,
	reco::CandSecondaryVertexTagInfo, svTagInfos,
	reco::CandSoftLeptonTagInfo, muonTagInfos,
	reco::CandSoftLeptonTagInfo, elecTagInfos> CandidateCombinedSVSoftLeptonCvsLJetTagComputer;

typedef JetTagComputerESProducer<CandidateCombinedSVJetTagComputer> CandidateCombinedSecondaryVertexESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CandidateCombinedSecondaryVertexESProducer);

typedef JetTagComputerESProducer<CombinedSVJetTagComputer> CombinedSecondaryVertexESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CombinedSecondaryVertexESProducer);

typedef JetTagComputerESProducer<CandidateCombinedSVSoftLeptonJetTagComputer> CandidateCombinedSecondaryVertexSoftLeptonESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CandidateCombinedSecondaryVertexSoftLeptonESProducer);

typedef JetTagComputerESProducer<CombinedSVSoftLeptonJetTagComputer> CombinedSecondaryVertexSoftLeptonESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CombinedSecondaryVertexSoftLeptonESProducer);

typedef JetTagComputerESProducer<GhostTrackJetTagComputer> GhostTrackESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(GhostTrackESProducer);

typedef JetTagComputerESProducer<CandidateSimpleSecondaryVertexComputer> CandidateSimpleSecondaryVertexESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CandidateSimpleSecondaryVertexESProducer);

typedef JetTagComputerESProducer<SimpleSecondaryVertexComputer> SimpleSecondaryVertexESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(SimpleSecondaryVertexESProducer);

typedef JetTagComputerESProducer<CandidateCombinedSVSoftLeptonJetTagComputer> CandidateCombinedSecondaryVertexSoftLeptonCvsLESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CandidateCombinedSecondaryVertexSoftLeptonCvsLESProducer);

typedef JetTagComputerESProducer<CandidateBoostedDoubleSecondaryVertexComputer> CandidateBoostedDoubleSecondaryVertexESProducer;
DEFINE_FWK_EVENTSETUP_MODULE(CandidateBoostedDoubleSecondaryVertexESProducer);
