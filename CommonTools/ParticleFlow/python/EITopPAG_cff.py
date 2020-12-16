import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import *
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
from RecoBTag.SecondaryVertex.candidateCombinedSecondaryVertexV2Computer_cfi import candidateCombinedSecondaryVertexV2Computer
from RecoBTag.SecondaryVertex.pfCombinedInclusiveSecondaryVertexV2BJetTags_cfi import pfCombinedInclusiveSecondaryVertexV2BJetTags


#### PU Again... need to do this twice because the "linking" stage of PF reco ####
#### condenses information into the new "particleFlow" collection.            ####


pfPileUpEI   = pfPileUp.clone( PFCandidates = 'particleFlowPtrs' )
pfNoPileUpEI = pfNoPileUp.clone( bottomCollection = 'particleFlowPtrs',
                                 topCollection = 'pfPileUpEI' )

pfPileUpJMEEI   = pfPileUpJME.clone( PFCandidates = 'particleFlowPtrs' )
pfNoPileUpJMEEI = pfNoPileUpJME.clone( bottomCollection = 'particleFlowPtrs',
                                       topCollection = 'pfPileUpJMEEI' )


#### Muons ####

pfAllMuonsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("pfNoPileUpEI"),
    cut = cms.string("abs(pdgId())==13"
        ),
    makeClones = cms.bool(True)
)

pfMuonsFromVertexEI = pfMuonsFromVertex.clone( src = 'pfAllMuonsEI' )

pfIsolatedMuonsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("pfMuonsFromVertexEI"),
    cut = cms.string('''abs(eta)<2.5 && pt>10. && muonRef.isAvailable() &&
    (muonRef.pfIsolationR04().sumChargedHadronPt+
    max(0.,muonRef.pfIsolationR04().sumNeutralHadronEt+
    muonRef.pfIsolationR04().sumPhotonEt-
    0.50*muonRef.pfIsolationR04().sumPUPt))/pt < 0.20 &&
    (muonRef.isPFMuon && (muonRef.isGlobalMuon || muonRef.isTrackerMuon) )'''
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
    cut = cms.string("abs(pdgId())==11"
        ),
    makeClones = cms.bool(True)
)

pfElectronsFromVertexEI = pfElectronsFromVertex.clone( src = 'pfAllElectronsEI' )

pfIsolatedElectronsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("pfElectronsFromVertexEI"),
    cut = cms.string('''abs(eta)<2.5 && pt>20. &&
    gsfTrackRef.isAvailable() &&
    gsfTrackRef.hitPattern().numberOfLostHits('MISSING_INNER_HITS')<2 &&
    (gsfElectronRef.pfIsolationVariables().sumChargedHadronPt+
    max(0.,gsfElectronRef.pfIsolationVariables().sumNeutralHadronEt+
    gsfElectronRef.pfIsolationVariables().sumPhotonEt-
    0.5*gsfElectronRef.pfIsolationVariables().sumPUPt))/pt < 0.15
    '''),
    makeClones = cms.bool(True)
)


pfNoElectron.topCollection    = 'pfIsolatedElectronsEI'
pfNoElectron.bottomCollection = 'pfNoMuon'

pfNoElectronJME.topCollection    = 'pfIsolatedElectronsEI'
pfNoElectronJME.bottomCollection = 'pfNoMuonJME'


#### Jets ####

pfJetsEI = pfJets.clone()
pfJetsPtrsEI = pfJetsPtrs.clone( src = "pfJetsEI" )

pfJetSequenceEI = cms.Sequence( pfJetsEI+ pfJetsPtrsEI )

pfNoJetEI = pfNoJet.clone(
    topCollection = 'pfJetsPtrsEI',
    bottomCollection = 'pfNoElectronJME'
)

#### Taus ####
pfTausEI = pfTaus.clone()
pfTausPtrsEI = pfTausPtrs.clone( src = "pfTausEI" )
pfNoTauEI = pfNoTau.clone(
    topCollection = 'pfTausPtrsEI',
    bottomCollection = 'pfJetsPtrsEI'
)

pfTauEISequence = cms.Sequence(
    pfTausPreSequence+
    pfTausBaseSequence+
    pfTausEI+
    pfTausPtrsEI
    )

#### B-tagging ####
pfImpactParameterTagInfosEI = pfImpactParameterTagInfos.clone(
    jets =  'pfJetsEI'
)
pfInclusiveSecondaryVertexFinderTagInfosEI = pfInclusiveSecondaryVertexFinderTagInfos.clone(
    trackIPTagInfos =  'pfImpactParameterTagInfosEI'
)
pfCombinedInclusiveSecondaryVertexV2BJetTagsEI = pfCombinedInclusiveSecondaryVertexV2BJetTags.clone(
    tagInfos = ["pfImpactParameterTagInfosEI",
                "pfInclusiveSecondaryVertexFinderTagInfosEI"]
)



#### MET ####
pfMetEI = pfMET.clone(srcJets="pfJetsEI")

#EITopPAG = cms.Sequence(
EIsequence = cms.Sequence(
    goodOfflinePrimaryVertices +
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

