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
    cut = cms.string("pt > 5 & muonRef.isAvailable() & "\
                     "muonRef.pfIsolationR04().sumChargedHadronPt + "\
                     "muonRef.pfIsolationR04().sumNeutralHadronEt + "\
                     "muonRef.pfIsolationR04().sumPhotonEt "\
                     " < 0.15 * pt "
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
    cut = cms.string(" pt > 5 & gsfElectronRef.isAvailable() & gsfTrackRef.trackerExpectedHitsInner.numberOfLostHits<2 & "\
                     "gsfElectronRef.pfIsolationVariables().chargedHadronIso + "\
                     "gsfElectronRef.pfIsolationVariables().neutralHadronIso + "\
                     "gsfElectronRef.pfIsolationVariables().photonIso "\
                     " < 0.2 * pt "
        ),
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

#### MET ####
pfMetEI = pfMET.clone(jets=cms.InputTag("pfJetsEI"))

EITopPAG = cms.Sequence(
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
    pfMetEI
    )

