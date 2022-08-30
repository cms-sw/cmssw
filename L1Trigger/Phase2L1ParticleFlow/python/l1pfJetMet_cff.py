import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.L1SeedConePFJetProducer_cfi import l1tSeedConePFJetProducer, l1tSeedConePFJetEmulatorProducer
from L1Trigger.Phase2L1ParticleFlow.DeregionizerProducer_cfi import l1tDeregionizerProducer as l1tLayer2Deregionizer
scPFL1PF            = l1tSeedConePFJetProducer.clone(L1PFObjects = 'l1tLayer1:PF')
scPFL1Puppi         = l1tSeedConePFJetProducer.clone()
scPFL1PuppiEmulator = l1tSeedConePFJetEmulatorProducer.clone(L1PFObject = cms.InputTag('l1tLayer2Deregionizer:Puppi'))

correctedJets = cms.EDProducer("L1TCorrectedPFJetProducer", 
    jets = cms.InputTag("_tag_"),
    correctorFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs.PU200_110X.root"),
    correctorDir = cms.string("_dir_")
)
# Using phase2_hgcalV10 to customize the config for all 106X samples, since there's no other modifier for it
from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
phase2_hgcalV10.toModify(correctedJets, correctorFile = "L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs.PU200_106X.root")
from Configuration.Eras.Modifier_phase2_hgcalV11_cff import phase2_hgcalV11
phase2_hgcalV11.toModify(correctedJets, correctorFile = "L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs.PU200_110X.root")

scPFL1PuppiCorrectedEmulator = correctedJets.clone(jets = 'scPFL1PuppiEmulator', correctorDir = 'L1PuppiSC4EmuDeregJets')

from L1Trigger.Phase2L1ParticleFlow.L1MhtPfProducer_cfi import l1tMHTPFProducer
scPFL1PuppiCorrectedEmulatorMHT = l1tMHTPFProducer.clone() 

L1TPFJetsTask = cms.Task(
    l1tLayer2Deregionizer, scPFL1PF, scPFL1Puppi, scPFL1PuppiEmulator, scPFL1PuppiCorrectedEmulator, scPFL1PuppiCorrectedEmulatorMHT
)


