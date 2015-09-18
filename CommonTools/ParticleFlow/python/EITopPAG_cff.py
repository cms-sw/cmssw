import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfMET_cfi  import *
from CommonTools.ParticleFlow.pfJets_cff import *
from CommonTools.ParticleFlow.pfTaus_cff import *
from CommonTools.ParticleFlow.ParticleSelectors.pfMuonsFromVertex_cfi import pfMuonsFromVertex
from CommonTools.ParticleFlow.ParticleSelectors.pfElectronsFromVertex_cfi import pfElectronsFromVertex
from CommonTools.ParticleFlow.pfNoPileUp_cff  import *
from CommonTools.ParticleFlow.pfNoPileUpJME_cff  import *


# sequential top projection cleaning
from CommonTools.ParticleFlow.TopProjectors.pfNoMuon_cfi import *
from CommonTools.ParticleFlow.TopProjectors.pfNoElectron_cfi import *
from CommonTools.ParticleFlow.TopProjectors.pfNoJet_cfi import *
from CommonTools.ParticleFlow.TopProjectors.pfNoTau_cfi import *


# b-tagging
from RecoBTag.ImpactParameter.pfImpactParameterTagInfos_cfi import pfImpactParameterTagInfos
from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi import pfInclusiveSecondaryVertexFinderTagInfos
from RecoBTag.SecondaryVertex.pfCombinedInclusiveSecondaryVertexV2BJetTags_cfi import pfCombinedInclusiveSecondaryVertexV2BJetTags


#### PU Again... need to do this twice because the "linking" stage of PF reco ####
#### condenses information into the new "particleFlow" collection.            ####


pfPileUpEI = pfPileUp.clone( PFCandidates = cms.InputTag('particleFlowPtrs') )
pfNoPileUpEI = pfNoPileUp.clone( bottomCollection = cms.InputTag('particleFlowPtrs'),
                                 topCollection = cms.InputTag('pfPileUpEI') )

pfPileUpJMEEI = pfPileUpJME.clone( PFCandidates = cms.InputTag('particleFlowPtrs') )
pfNoPileUpJMEEI = pfNoPileUpJME.clone( bottomCollection = cms.InputTag('particleFlowPtrs'),
                                       topCollection = cms.InputTag('pfPileUpJMEEI') )


#### Muons ####

pfAllMuonsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("pfNoPileUpEI"),
    cut = cms.string("std::abs(obj.pdgId())==13"
        ),
    makeClones = cms.bool(True)
)

pfMuonsFromVertexEI = pfMuonsFromVertex.clone( src = cms.InputTag('pfAllMuonsEI') )

pfIsolatedMuonsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("pfMuonsFromVertexEI"),
    cut = cms.string('''std::abs(obj.eta())<2.5 && obj.pt()>10. && obj.muonRef().isAvailable() &&
    (obj.muonRef()->pfIsolationR04().sumChargedHadronPt+
    std::max(0.,obj.muonRef()->pfIsolationR04().sumNeutralHadronEt+
    obj.muonRef()->pfIsolationR04().sumPhotonEt-
    0.50*obj.muonRef()->pfIsolationR04().sumPUPt))/obj.pt() < 0.20 &&
    (obj.muonRef()->isPFMuon() && (obj.muonRef()->isGlobalMuon() || obj.muonRef()->isTrackerMuon()) )'''
    ),
    makeClones = cms.bool(True)
)



pfNoMuon.topCollection    = 'pfIsolatedMuonsEI'
pfNoMuon.bottomCollection = 'pfNoPileUpEI'


pfNoMuonJME.topCollection    = 'pfIsolatedMuonsEI'
pfNoMuonJME.bottomCollection = 'pfNoPileUpJMEEI'



#### Electrons ####

pfAllElectronsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("pfNoMuon"),
    cut = cms.string("std::abs(obj.pdgId())==11"
        ),
    makeClones = cms.bool(True)
)

pfElectronsFromVertexEI = pfElectronsFromVertex.clone( src = cms.InputTag('pfAllElectronsEI') )

pfIsolatedElectronsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("pfElectronsFromVertexEI"),
    cut = cms.string('''std::abs(obj.eta())<2.5 && obj.pt()>20. &&
    obj.gsfTrackRef().isAvailable() &&
    obj.gsfTrackRef()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS)<2 &&
    (obj.gsfElectronRef()->pfIsolationVariables().sumChargedHadronPt+
    std::max(0.,obj.gsfElectronRef()->pfIsolationVariables().sumNeutralHadronEt+
    obj.gsfElectronRef()->pfIsolationVariables().sumPhotonEt-
    0.5*obj.gsfElectronRef()->pfIsolationVariables().sumPUPt))/obj.pt() < 0.15
    '''),
    makeClones = cms.bool(True)
)


pfNoElectron.topCollection    = 'pfIsolatedElectronsEI'
pfNoElectron.bottomCollection = 'pfNoMuon'

pfNoElectronJME.topCollection    = 'pfIsolatedElectronsEI'
pfNoElectronJME.bottomCollection = 'pfNoMuonJME'


#### Jets ####

pfJetsEI = pfJets.clone()
pfJetsPtrsEI = pfJetsPtrs.clone(src=cms.InputTag("pfJetsEI"))

pfJetSequenceEI = cms.Sequence( pfJetsEI+ pfJetsPtrsEI )

pfNoJetEI = pfNoJet.clone(
    topCollection = 'pfJetsPtrsEI',
    bottomCollection = 'pfNoElectronJME'
    )

#### Taus ####
pfTausEI = pfTaus.clone()
pfTausPtrsEI = pfTausPtrs.clone(src=cms.InputTag("pfTausEI") )
pfNoTauEI = pfNoTau.clone(
    topCollection = cms.InputTag('pfTausPtrsEI'),
    bottomCollection = cms.InputTag('pfJetsPtrsEI')
    )

pfTauEISequence = cms.Sequence(
    pfTausPreSequence+
    pfTausBaseSequence+
    pfTausEI+
    pfTausPtrsEI
    )

#### B-tagging ####
pfImpactParameterTagInfosEI = pfImpactParameterTagInfos.clone(
    jets = cms.InputTag( 'pfJetsEI' )
    )
pfInclusiveSecondaryVertexFinderTagInfosEI = pfInclusiveSecondaryVertexFinderTagInfos.clone(
    trackIPTagInfos = cms.InputTag( 'pfImpactParameterTagInfosEI' )
    )
pfCombinedInclusiveSecondaryVertexV2BJetTagsEI = pfCombinedInclusiveSecondaryVertexV2BJetTags.clone(
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfosEI"),
                             cms.InputTag("pfInclusiveSecondaryVertexFinderTagInfosEI"))
    )



#### MET ####
pfMetEI = pfMET.clone(jets=cms.InputTag("pfJetsEI"))

#EITopPAG = cms.Sequence(
EIsequence = cms.Sequence(
    pfPileUpEI +
    pfPileUpJMEEI +
    pfNoPileUpEI +
    pfNoPileUpJMEEI +
    pfAllMuonsEI +
    pfMuonsFromVertexEI +
    pfIsolatedMuonsEI +
    pfNoMuon +
    pfNoMuonJME +
    pfAllElectronsEI +
    pfElectronsFromVertexEI +
    pfIsolatedElectronsEI +
    pfNoElectron +
    pfNoElectronJME +
    pfJetSequenceEI +
    pfNoJetEI +
    pfTauEISequence +
    pfNoTauEI +
    pfMetEI+
    pfImpactParameterTagInfosEI+
    pfInclusiveSecondaryVertexFinderTagInfosEI+
    pfCombinedInclusiveSecondaryVertexV2BJetTagsEI
    )

