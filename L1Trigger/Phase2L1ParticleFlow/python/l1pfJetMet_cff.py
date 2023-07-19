import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1ParticleFlow.l1tSeedConePFJetProducer_cfi import l1tSeedConePFJetProducer, l1tSeedConePFJetEmulatorProducer
from L1Trigger.Phase2L1ParticleFlow.l1tDeregionizerProducer_cfi import l1tDeregionizerProducer as l1tLayer2Deregionizer, l1tDeregionizerProducerExtended as l1tLayer2DeregionizerExtended
l1tSC4PFL1PF            = l1tSeedConePFJetProducer.clone(L1PFObjects = 'l1tLayer1:PF')
l1tSC4PFL1Puppi         = l1tSeedConePFJetProducer.clone()
l1tSC4PFL1PuppiEmulator = l1tSeedConePFJetEmulatorProducer.clone(L1PFObjects = 'l1tLayer2Deregionizer:Puppi')
l1tSC8PFL1PuppiEmulator = l1tSeedConePFJetEmulatorProducer.clone(L1PFObjects = 'l1tLayer2Deregionizer:Puppi',
                                                                 coneSize = cms.double(0.8))
l1tSC4PFL1PuppiCorrectedEmulator = l1tSeedConePFJetEmulatorProducer.clone(L1PFObjects = 'l1tLayer2Deregionizer:Puppi',
                                                                          doCorrections = cms.bool(True),
                                                                          correctorFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs_20220308.root"),
                                                                          correctorDir = cms.string('L1PuppiSC4EmuJets'))
l1tSC8PFL1PuppiCorrectedEmulator = l1tSeedConePFJetEmulatorProducer.clone(L1PFObjects = 'l1tLayer2Deregionizer:Puppi',
                                                                          coneSize = cms.double(0.8),
                                                                          doCorrections = cms.bool(True),
                                                                          correctorFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs_20220308.root"),
                                                                          correctorDir = cms.string('L1PuppiSC4EmuJets'))

_correctedJets = cms.EDProducer("L1TCorrectedPFJetProducer", 
    jets = cms.InputTag("_tag_"),
    correctorFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs_20220308.root"),
    correctorDir = cms.string("_dir_"),
    copyDaughters = cms.bool(False),
    emulate = cms.bool(False)
)

# Using phase2_hgcalV10 to customize the config for all 106X samples, since there's no other modifier for it
from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
phase2_hgcalV10.toModify(_correctedJets, correctorFile = "L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs.PU200_106X.root")
from Configuration.Eras.Modifier_phase2_hgcalV11_cff import phase2_hgcalV11
phase2_hgcalV11.toModify(_correctedJets, correctorFile = "L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs_20220308.root")

from L1Trigger.Phase2L1ParticleFlow.l1tMHTPFProducer_cfi import l1tMHTPFProducer
l1tSC4PFL1PuppiCorrectedEmulatorMHT = l1tMHTPFProducer.clone(jets = 'l1tSC4PFL1PuppiCorrectedEmulator')

l1tSC4PFL1PuppiExtended         = l1tSeedConePFJetProducer.clone(L1PFObjects = 'l1tLayer1Extended:Puppi')
l1tSC4PFL1PuppiExtendedEmulator = l1tSeedConePFJetEmulatorProducer.clone(L1PFObjects = 'l1tLayer2DeregionizerExtended:Puppi')
l1tSC4PFL1PuppiExtendedCorrectedEmulator = l1tSeedConePFJetEmulatorProducer.clone(L1PFObjects = 'l1tLayer2DeregionizerExtended:Puppi',
                                                                     doCorrections = cms.bool(True),
                                                                     correctorFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs_20220308.root"),
                                                                     correctorDir = cms.string('L1PuppiSC4EmuJets'))

L1TPFJetsTask = cms.Task(
    l1tLayer2Deregionizer, l1tSC4PFL1PF, l1tSC4PFL1Puppi, l1tSC4PFL1PuppiEmulator, l1tSC4PFL1PuppiCorrectedEmulator, l1tSC4PFL1PuppiCorrectedEmulatorMHT,
    l1tSC8PFL1PuppiEmulator, l1tSC8PFL1PuppiCorrectedEmulator
)

L1TPFJetsExtendedTask = cms.Task(
    l1tLayer2DeregionizerExtended, l1tSC4PFL1PuppiExtended, l1tSC4PFL1PuppiExtendedEmulator, l1tSC4PFL1PuppiExtendedCorrectedEmulator
)

L1TPFJetsEmulationTask = cms.Task(
    l1tLayer2Deregionizer, l1tSC4PFL1PuppiEmulator, l1tSC4PFL1PuppiCorrectedEmulator, l1tSC4PFL1PuppiCorrectedEmulatorMHT,
    l1tSC8PFL1PuppiEmulator, l1tSC8PFL1PuppiCorrectedEmulator
)
