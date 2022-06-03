import FWCore.ParameterSet.Config as cms

from RecoMET.METProducers.pfMet_cfi import pfMet
_pfMet         =  pfMet.clone(calculateSignificance = False)
l1PFMetCalo    = _pfMet.clone(src = "l1pfCandidates:Calo")
l1PFMetPF      = _pfMet.clone(src = "l1pfCandidates:PF")
l1PFMetPuppi   = _pfMet.clone(src = "l1pfCandidates:Puppi")

l1PFMetsTask = cms.Task(l1PFMetCalo, l1PFMetPF, l1PFMetPuppi)

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
_ak4PFJets      =  ak4PFJets.clone(doAreaFastjet = False)
ak4PFL1Calo    = _ak4PFJets.clone(src = 'l1pfCandidates:Calo')
ak4PFL1PF      = _ak4PFJets.clone(src = 'l1pfCandidates:PF')
ak4PFL1Puppi   = _ak4PFJets.clone(src = 'l1pfCandidates:Puppi')

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
        
ak4PFL1CaloCorrected = _correctedJets.clone(jets = 'ak4PFL1Calo', correctorDir = 'L1CaloJets')
ak4PFL1PFCorrected = _correctedJets.clone(jets = 'ak4PFL1PF', correctorDir = 'L1PFJets')
ak4PFL1PuppiCorrected = _correctedJets.clone(jets = 'ak4PFL1Puppi', correctorDir = 'L1PuppiJets')

scPFL1PuppiCorrectedEmulator = _correctedJets.clone(jets = 'scPFL1PuppiEmulator', correctorDir = 'L1PuppiSC4EmuDeregJets')

l1PFJetsTask = cms.Task(
    ak4PFL1Calo, ak4PFL1PF, ak4PFL1Puppi,
    ak4PFL1CaloCorrected, ak4PFL1PFCorrected, ak4PFL1PuppiCorrected,
    l1ctLayer2Deregionizer, scPFL1PF, scPFL1Puppi, scPFL1PuppiEmulator, scPFL1PuppiCorrectedEmulator
)


