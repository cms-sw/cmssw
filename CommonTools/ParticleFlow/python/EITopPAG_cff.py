import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.pfMET_cfi  import *
from CommonTools.ParticleFlow.pfJets_cff import *
from CommonTools.ParticleFlow.pfTaus_cff import *

# sequential top projection cleaning
from CommonTools.ParticleFlow.TopProjectors.pfNoMuon_cfi import * 
from CommonTools.ParticleFlow.TopProjectors.pfNoElectron_cfi import * 
from CommonTools.ParticleFlow.TopProjectors.pfNoJet_cfi import *
from CommonTools.ParticleFlow.TopProjectors.pfNoTau_cfi import *

pfIsolatedMuonsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("particleFlow"),
    cut = cms.string("abs(pdgId())==13 & muonRef.isAvailable() & "\
                     "muonRef.pfIsolationR03().sumChargedHadronPt + "\
                     "muonRef.pfIsolationR03().sumNeutralHadronEt + "\
                     "muonRef.pfIsolationR03().sumPhotonEt "\
                     " < 0.2 * pt "
        ),
    makeClones = cms.bool(True)
)


pfIsolatedElectronsEI = cms.EDFilter(
    "PFCandidateFwdPtrCollectionStringFilter",
    src = cms.InputTag("particleFlow"),
    cut = cms.string("abs(pdgId())==11 & gsfElectronRef.isAvailable() & "\
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
    pfIsolatedMuonsEI +
    pfNoMuon +
    pfNoMuonJME +
    pfIsolatedElectronsEI +    
    pfNoElectron +
    pfNoElectronJME +
    pfJetSequence +
    pfNoJet + 
    pfTauSequence +
    pfNoTau +
    pfMET
    )

