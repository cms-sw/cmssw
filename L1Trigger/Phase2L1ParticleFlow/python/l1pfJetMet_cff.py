import FWCore.ParameterSet.Config as cms

from RecoMET.METProducers.pfMet_cfi import pfMet
_pfMet         =  pfMet.clone(calculateSignificance = False)
l1PFMetCalo    = _pfMet.clone(src = "l1pfCandidates:Calo")
l1PFMetPF      = _pfMet.clone(src = "l1pfCandidates:PF")
l1PFMetPuppi   = _pfMet.clone(src = "l1pfCandidates:Puppi")

l1PFMets = cms.Sequence(l1PFMetCalo + l1PFMetPF + l1PFMetPuppi)

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
_ak4PFJets      =  ak4PFJets.clone(doAreaFastjet = False)
ak4PFL1Calo    = _ak4PFJets.clone(src = 'l1pfCandidates:Calo')
ak4PFL1PF      = _ak4PFJets.clone(src = 'l1pfCandidates:PF')
ak4PFL1Puppi   = _ak4PFJets.clone(src = 'l1pfCandidates:Puppi')

_correctedJets = cms.EDProducer("L1TCorrectedPFJetProducer", 
    jets = cms.InputTag("_tag_"),
    correctorFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs.PU200.root"),
    correctorDir = cms.string("_dir_"),
    copyDaughters = cms.bool(False)
)
# Using phase2_hgcalV10 to customize the config for all 106X samples, since there's no other modifier for it
from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
phase2_hgcalV10.toModify(_correctedJets, correctorFile = "L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs.PU200_106X.root")
from Configuration.Eras.Modifier_phase2_hgcalV11_cff import phase2_hgcalV11
phase2_hgcalV11.toModify(_correctedJets, correctorFile = "L1Trigger/Phase2L1ParticleFlow/data/jecs/jecs.PU200_110X.root")
        
ak4PFL1CaloCorrected = _correctedJets.clone(jets = 'ak4PFL1Calo', correctorDir = 'L1CaloJets')
ak4PFL1PFCorrected = _correctedJets.clone(jets = 'ak4PFL1PF', correctorDir = 'L1PFJets')
ak4PFL1PuppiCorrected = _correctedJets.clone(jets = 'ak4PFL1Puppi', correctorDir = 'L1PuppiJets')

l1PFJets = cms.Sequence(
    ak4PFL1Calo + ak4PFL1PF + ak4PFL1Puppi +
    ak4PFL1CaloCorrected + ak4PFL1PFCorrected + ak4PFL1PuppiCorrected
)


