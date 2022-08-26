import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.l1tSeedConePFJetProducer_cfi import l1tSeedConePFJetProducer, l1tSeedConePFJetEmulatorProducer
from L1Trigger.Phase2L1ParticleFlow.l1tDeregionizerProducer_cfi import l1tDeregionizerProducer as l1tLayer2Deregionizer, l1tDeregionizerProducerExtended as l1tLayer2DeregionizerExtended
l1tSCPFL1PF            = l1tSeedConePFJetProducer.clone(L1PFObjects = 'l1tLayer1:PF')
l1tSCPFL1Puppi         = l1tSeedConePFJetProducer.clone()
l1tSCPFL1PuppiEmulator = l1tSeedConePFJetEmulatorProducer.clone(L1PFObjects = 'l1tLayer2Deregionizer:Puppi')

_correctedJets = cms.EDProducer("L1TCorrectedPFJetProducer", 
    jets = cms.InputTag("_tag_"),
    correctorFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs.PU200_110X.root"),
    correctorDir = cms.string("_dir_")
)
# Using phase2_hgcalV10 to customize the config for all 106X samples, since there's no other modifier for it
from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
phase2_hgcalV10.toModify(_correctedJets, correctorFile = "L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs.PU200_106X.root")
from Configuration.Eras.Modifier_phase2_hgcalV11_cff import phase2_hgcalV11
phase2_hgcalV11.toModify(_correctedJets, correctorFile = "L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs.PU200_110X.root")

l1tSCPFL1PuppiCorrectedEmulator = _correctedJets.clone(jets = 'l1tSCPFL1PuppiEmulator', correctorDir = 'L1PuppiSC4EmuDeregJets')

from L1Trigger.Phase2L1ParticleFlow.l1tMHTPFProducer_cfi import l1tMHTPFProducer
l1tSCPFL1PuppiCorrectedEmulatorMHT = l1tMHTPFProducer.clone() 

L1TPFJetsTask = cms.Task(
    l1tLayer2Deregionizer, l1tSCPFL1PF, l1tSCPFL1Puppi, l1tSCPFL1PuppiEmulator, l1tSCPFL1PuppiCorrectedEmulator, l1tSCPFL1PuppiCorrectedEmulatorMHT
)

l1tSCPFL1PuppiExtended         = l1tSeedConePFJetProducer.clone(L1PFObjects = 'l1tLayer1Extended:Puppi')
l1tSCPFL1PuppiExtendedEmulator = l1tSeedConePFJetEmulatorProducer.clone(L1PFObjects = 'l1tLayer2DeregionizerExtended:Puppi')
l1tSCPFL1PuppiExtendedCorrectedEmulator = _correctedJets.clone(jets = 'l1tSCPFL1PuppiExtendedEmulator', correctorDir = 'L1PuppiSC4EmuDeregJets')

L1TPFJetsTask = cms.Task(
    l1tLayer2Deregionizer, l1tSCPFL1PF, l1tSCPFL1Puppi, l1tSCPFL1PuppiEmulator, l1tSCPFL1PuppiCorrectedEmulator, l1tSCPFL1PuppiCorrectedEmulatorMHT
)

L1PFJetsExtendedTask = cms.Task(
    l1tLayer2DeregionizerExtended, l1tSCPFL1PuppiExtended, l1tSCPFL1PuppiExtendedEmulator, l1tSCPFL1PuppiExtendedCorrectedEmulator
)

L1TPFJetsEmulationTask = cms.Task(
    l1tLayer2Deregionizer, l1tSCPFL1PuppiEmulator, l1tSCPFL1PuppiCorrectedEmulator, l1tSCPFL1PuppiCorrectedEmulatorMHT
)

