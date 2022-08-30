import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.L1SeedConePFJetProducer_cfi import L1SeedConePFJetProducer, L1SeedConePFJetEmulatorProducer
from L1Trigger.Phase2L1ParticleFlow.DeregionizerProducer_cfi import DeregionizerProducer as l1ctLayer2Deregionizer
scPFL1PF            = L1SeedConePFJetProducer.clone(L1PFObjects = 'l1ctLayer1:PF')
scPFL1Puppi         = L1SeedConePFJetProducer.clone()
scPFL1PuppiEmulator = L1SeedConePFJetEmulatorProducer.clone(L1PFObject = cms.InputTag('l1ctLayer2Deregionizer:Puppi'))

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

scPFL1PuppiCorrectedEmulator = _correctedJets.clone(jets = 'scPFL1PuppiEmulator', correctorDir = 'L1PuppiSC4EmuDeregJets')

from L1Trigger.Phase2L1ParticleFlow.L1MhtPfProducer_cfi import L1MhtPfProducer
scPFL1PuppiCorrectedEmulatorMHT = L1MhtPfProducer.clone() 

l1PFJetsTask = cms.Task(
    l1ctLayer2Deregionizer, scPFL1PF, scPFL1Puppi, scPFL1PuppiEmulator, scPFL1PuppiCorrectedEmulator, scPFL1PuppiCorrectedEmulatorMHT
)


