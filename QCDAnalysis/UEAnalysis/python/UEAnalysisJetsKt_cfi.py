import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
FastjetWithAreaPU = cms.PSet(
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    UE_Subtraction = cms.string('no')
)
UEAnalysisKtJetParameters = cms.PSet(
    verbose = cms.untracked.bool(False),
    JetPtMin = cms.double(1.0),
    inputEtMin = cms.double(0.9),
    FJ_ktRParam = cms.double(0.4),
    jetType = cms.untracked.string('GenJet'),
    inputEMin = cms.double(0.0)
)
ueKt4ChgGenJet = cms.EDProducer("KtJetProducer",
    UEAnalysisKtJetParameters,
    KtJetParameters,
    FastjetNoPU,
    src = cms.InputTag("chargeParticles")
)
ueKt4TracksJet = cms.EDProducer("KtJetProducer",
                                UEAnalysisKtJetParameters,
                                KtJetParameters,
                                FastjetNoPU,
                                src = cms.InputTag("goodTracks")
                                )
ueKt4TracksJet.jetType = 'BasicJet'


UEAnalysisJetsKtOnlyMC = cms.Sequence(ueKt4ChgGenJet)
UEAnalysisJetsKtOnlyReco = cms.Sequence(ueKt4TracksJet)
UEAnalysisJetsKt = cms.Sequence(UEAnalysisJetsKtOnlyMC*UEAnalysisJetsKtOnlyReco)


