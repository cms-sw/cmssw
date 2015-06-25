import FWCore.ParameterSet.Config as cms

from L1Trigger.Configuration.ValL1Emulator_cff import *

# L1 emulator tree
import L1TriggerDPG.L1Ntuples.l1NtupleProducer_cfi
l1EmulatorTree = L1TriggerDPG.L1Ntuples.l1NtupleProducer_cfi.l1NtupleProducer.clone()
l1EmulatorTree.generatorSource      = cms.InputTag("none")
l1EmulatorTree.simulationSource     = cms.InputTag("none")
l1EmulatorTree.hltSource            = cms.InputTag("none")
l1EmulatorTree.gmtSource            = cms.InputTag("valGmtDigis")
l1EmulatorTree.gtEvmSource          = cms.InputTag("none")
l1EmulatorTree.gtSource             = cms.InputTag("valGtDigis")
l1EmulatorTree.gctCentralJetsSource = cms.InputTag("valGctDigis","cenJets")
l1EmulatorTree.gctNonIsoEmSource    = cms.InputTag("valGctDigis","nonIsoEm")
l1EmulatorTree.gctForwardJetsSource = cms.InputTag("valGctDigis","forJets")
l1EmulatorTree.gctIsoEmSource       = cms.InputTag("valGctDigis","isoEm")
l1EmulatorTree.gctEnergySumsSource  = cms.InputTag("valGctDigis","")
l1EmulatorTree.gctTauJetsSource     = cms.InputTag("valGctDigis","tauJets")
l1EmulatorTree.rctSource            = cms.InputTag("valRctDigis")
l1EmulatorTree.dttfSource           = cms.InputTag("valDttfDigis")
l1EmulatorTree.csctfTrkSource       = cms.InputTag("valCsctfDigis")	
l1EmulatorTree.csctfLCTSource       = cms.InputTag("valCsctfDigis") 
l1EmulatorTree.csctfStatusSource    = cms.InputTag("valCsctfDigis")
l1EmulatorTree.csctfDTStubsSource   = cms.InputTag("valCsctfDigis:DT")
l1EmulatorTree.ecalSource           = cms.InputTag("none")
l1EmulatorTree.hcalSource           = cms.InputTag("none")

# L1Extra
import L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi
valL1extraParticles = L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi.l1extraParticles.clone()
valL1extraParticles.muonSource = cms.InputTag("valGmtDigis")
valL1extraParticles.nonIsolatedEmSource = cms.InputTag("valGctDigis","nonIsoEm")
valL1extraParticles.isolatedEmSource = cms.InputTag("valGctDigis","isoEm")
valL1extraParticles.forwardJetSource = cms.InputTag("valGctDigis","forJets")
valL1extraParticles.centralJetSource = cms.InputTag("valGctDigis","cenJets")
valL1extraParticles.tauJetSource = cms.InputTag("valGctDigis","tauJets")
valL1extraParticles.etMissSource = cms.InputTag("valGctDigis")
valL1extraParticles.htMissSource = cms.InputTag("valGctDigis")
valL1extraParticles.etTotalSource = cms.InputTag("valGctDigis")
valL1extraParticles.etHadSource = cms.InputTag("valGctDigis")
valL1extraParticles.hfRingEtSumsSource = cms.InputTag("valGctDigis")
valL1extraParticles.hfRingBitCountsSource = cms.InputTag("valGctDigis")

# L1Extra Tree
import L1TriggerDPG.L1Ntuples.l1ExtraTreeProducer_cfi
l1EmulatorExtraTree = L1TriggerDPG.L1Ntuples.l1ExtraTreeProducer_cfi.l1ExtraTreeProducer.clone()
l1EmulatorExtraTree.nonIsoEmLabel = cms.untracked.InputTag("valL1extraParticles:NonIsolated")
l1EmulatorExtraTree.isoEmLabel = cms.untracked.InputTag("valL1extraParticles:Isolated")
l1EmulatorExtraTree.tauJetLabel = cms.untracked.InputTag("valL1extraParticles:Tau")
l1EmulatorExtraTree.cenJetLabel = cms.untracked.InputTag("valL1extraParticles:Central")
l1EmulatorExtraTree.fwdJetLabel = cms.untracked.InputTag("valL1extraParticles:Forward")
l1EmulatorExtraTree.muonLabel = cms.untracked.InputTag("valL1extraParticles")
l1EmulatorExtraTree.metLabel = cms.untracked.InputTag("valL1extraParticles:MET")
l1EmulatorExtraTree.mhtLabel = cms.untracked.InputTag("valL1extraParticles:MHT")
l1EmulatorExtraTree.hfRingsLabel = cms.untracked.InputTag("valL1extraParticles")

# sequence
L1EmulatorTree = cms.Sequence(
    ValL1Emulator
    +l1EmulatorTree
    +valL1extraParticles
    +l1EmulatorExtraTree
)
