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
from RecoJets.JetAssociationProducers.ak4JTA_cff import ak4JetTracksAssociatorAtVertexPF
from RecoBTag.ImpactParameter.impactParameterTagInfos_cfi import impactParameterTagInfos
from RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi import secondaryVertexTagInfos
from RecoBTag.SecondaryVertex.combinedSecondaryVertexBJetTags_cfi import combinedSecondaryVertexBJetTags


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
    cut = cms.string("abs(pdgId())==13"
        ),
    makeClones = cms.bool(True)
)

pfMuonsFromVertexEI = pfMuonsFromVertex.clone( src = cms.InputTag('pfAllMuonsEI') )

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

pfElectronsFromVertexEI = pfElectronsFromVertex.clone( src = cms.InputTag('pfAllElectronsEI') )

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

ak4PFJetsCHSEI = jetAlgo('AK4')
ak4PFJetsCHSEIPtrs = pfJetsPtrs.clone(src=cms.InputTag("ak4PFJetsCHSEI"))

ak4PFJetsCHSEISequence = cms.Sequence( ak4PFJetsCHSEI + ak4PFJetsCHSEIPtrs )

pfNoJetEI = pfNoJet.clone(
    topCollection = 'ak4PFJetsCHSEIPtrs',
    bottomCollection = 'pfNoElectronJME'
    )

#### Taus ####
pfTausEI = pfTaus.clone()
pfTausPtrsEI = pfTausPtrs.clone(src=cms.InputTag("pfTausEI") )
pfNoTauEI = pfNoTau.clone(
    topCollection = cms.InputTag('pfTausPtrsEI'),
    bottomCollection = cms.InputTag('ak4PFJetsCHSEIPtrs')
    )

pfTauEISequence = cms.Sequence(
    pfTausPreSequence+
    pfTausBaseSequence+
    pfTausEI+
    pfTausPtrsEI
    )

#### B-tagging ####
ak4JetTracksAssociatorAtVertexPFEI = ak4JetTracksAssociatorAtVertexPF.clone (
    jets = cms.InputTag("ak4PFJetsCHSEI")
    )
impactParameterTagInfosEI = impactParameterTagInfos.clone(
    jetTracks = cms.InputTag( 'ak4JetTracksAssociatorAtVertexPFEI' )
    )
secondaryVertexTagInfosEI = secondaryVertexTagInfos.clone(
    trackIPTagInfos = cms.InputTag( 'impactParameterTagInfosEI' )
    )
combinedSecondaryVertexBJetTagsEI = combinedSecondaryVertexBJetTags.clone(
    tagInfos = cms.VInputTag(cms.InputTag("impactParameterTagInfosEI"),
                             cms.InputTag("secondaryVertexTagInfosEI"))
    )



#### MET ####
pfMetEI = pfMET.clone(jets=cms.InputTag("ak4PFJetsCHSEI"))

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
    ak4PFJetsCHSEISequence +
    pfNoJetEI +
    pfTauEISequence +
    pfNoTauEI +
    pfMetEI+
    ak4JetTracksAssociatorAtVertexPFEI+
    impactParameterTagInfosEI+
    secondaryVertexTagInfosEI+
    combinedSecondaryVertexBJetTagsEI
    )

