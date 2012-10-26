import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfMET_cfi  import *
from CommonTools.ParticleFlow.pfJets_cff import *
from CommonTools.ParticleFlow.pfTaus_cff import *
from CommonTools.ParticleFlow.ParticleSelectors.pfMuonsFromVertex_cfi import pfMuonsFromVertex
from CommonTools.ParticleFlow.ParticleSelectors.pfElectronsFromVertex_cfi import pfElectronsFromVertex

# sequential top projection cleaning
from CommonTools.ParticleFlow.TopProjectors.pfNoMuon_cfi import * 
from CommonTools.ParticleFlow.TopProjectors.pfNoElectron_cfi import * 
from CommonTools.ParticleFlow.TopProjectors.pfNoJet_cfi import *
from CommonTools.ParticleFlow.TopProjectors.pfNoTau_cfi import *



pfAllMuonsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("particleFlow"),
    cut = cms.string("abs(pdgId())==13"
        ),
    makeClones = cms.bool(True)
)


pfMuonsNoPUEI = cms.EDProducer(
    "TPPFCandidatesOnPFCandidates",
    enable =  cms.bool( True ),
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("muonsNoPUEI"),
    topCollection = cms.InputTag("pfPileUp"),
    bottomCollection = cms.InputTag("pfAllMuonsEI")
)

pfMuonsNoPUEIClones = cms.EDProducer("PFCandidateFromFwdPtrProducer",
                                     src=cms.InputTag('pfMuonsNoPUEI')
    )


pfMuonsFromVertexEI = pfMuonsFromVertex.clone( src = cms.InputTag('pfMuonsNoPUEIClones') )

pfIsolatedMuonsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("pfAllMuonsEI"),
    cut = cms.string("pt > 5 & muonRef.isAvailable() & "\
                     "muonRef.pfIsolationR03().sumChargedHadronPt + "\
                     "muonRef.pfIsolationR03().sumNeutralHadronEt + "\
                     "muonRef.pfIsolationR03().sumPhotonEt "\
                     " < 0.15 * pt "
        ),
    makeClones = cms.bool(True)
)


pfAllElectronsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("particleFlow"),
    cut = cms.string("abs(pdgId())==11"
        ),
    makeClones = cms.bool(True)
)


pfElectronsNoPUEI = cms.EDProducer(
    "TPPFCandidatesOnPFCandidates",
    enable =  cms.bool( True ),
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("muonsNoPUEI"),
    topCollection = cms.InputTag("pfPileUp"),
    bottomCollection = cms.InputTag("pfAllElectronsEI")
)


pfElectronsNoPUEIClones = cms.EDProducer("PFCandidateFromFwdPtrProducer",
                                     src=cms.InputTag('pfElectronsNoPUEI')
    )


pfElectronsFromVertexEI = pfElectronsFromVertex.clone( src = cms.InputTag('pfElectronsNoPUEIClones') )


pfIsolatedElectronsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("particleFlow"),
    cut = cms.string(" pt > 5 & gsfElectronRef.isAvailable() & gsfTrackRef.trackerExpectedHitsInner.numberOfLostHits<2 & "\
                     "gsfElectronRef.pfIsolationVariables().chargedHadronIso + "\
                     "gsfElectronRef.pfIsolationVariables().neutralHadronIso + "\
                     "gsfElectronRef.pfIsolationVariables().photonIso "\
                     " < 0.2 * pt "
        ),
    makeClones = cms.bool(True)
)


pfNoMuon.topCollection    = 'pfIsolatedMuonsEI'
pfNoMuon.bottomCollection = 'pfNoPileUp'

pfNoElectron.topCollection    = 'pfIsolatedElectronsEI'
pfNoElectron.bottomCollection = 'pfNoMuon'

pfNoMuonJME.topCollection    = 'pfIsolatedMuonsEI'
pfNoMuonJME.bottomCollection = 'pfNoPileUpJME'

pfNoElectronJME.topCollection    = 'pfIsolatedElectronsEI'
pfNoElectronJME.bottomCollection = 'pfNoMuonJME'

pfNoJet.topCollection = 'pfJetsPtrs'
pfNoJet.bottomCollection = 'pfNoElectronJME'

pfNoTau.topCollection = 'pfTausPtrs'
pfNoTau.bottomCollection = 'pfJetsPtrs'

EITopPAG = cms.Sequence(
    pfAllMuonsEI +
    pfMuonsNoPUEI +
    pfMuonsNoPUEIClones +
    pfMuonsFromVertexEI + 
    pfIsolatedMuonsEI +
    pfNoMuon +
    pfNoMuonJME +
    pfAllElectronsEI +
    pfElectronsNoPUEI +
    pfElectronsNoPUEIClones +
    pfElectronsFromVertexEI + 
    pfIsolatedElectronsEI +    
    pfNoElectron +
    pfNoElectronJME +
    pfJetSequence +
    pfNoJet + 
    pfTauSequence +
    pfNoTau +
    pfMET
    )

