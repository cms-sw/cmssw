import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.TrackJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

from RecoJets.JetProducers.kt4GenJets_cfi import kt4GenJets
from RecoJets.JetProducers.kt4TrackJets_cfi import kt4TrackJets

FastjetWithAreaPU = cms.PSet(
    Active_Area_Repeats = cms.int32(5),
    GhostArea = cms.double(0.01),
    Ghost_EtaMax = cms.double(6.0),
    UE_Subtraction = cms.string('no')
)


ueKt4ChgGenJet = kt4GenJets.clone(
      src = cms.InputTag("chargeParticles"),
      jetPtMin       = cms.double(1.0),
      inputEtMin     = cms.double(0.9)
)

ueKt4TracksJet =  kt4TrackJets.clone(
          src = cms.InputTag("goodTracks"),
jetPtMin       = cms.double(1.0),
inputEtMin     = cms.double(0.9)
)
ueKt4TracksJet.jetType = 'BasicJet'


UEAnalysisJetsKtOnlyMC = cms.Sequence(ueKt4ChgGenJet)
UEAnalysisJetsKtOnlyReco = cms.Sequence(ueKt4TracksJet)
UEAnalysisJetsKt = cms.Sequence(UEAnalysisJetsKtOnlyMC*UEAnalysisJetsKtOnlyReco)


