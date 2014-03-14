import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.packedPFCandidates_cfi import *
from PhysicsTools.PatAlgos.slimming.offlineSlimmedPrimaryVertices_cfi import *
from PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi import *
from PhysicsTools.PatAlgos.slimming.selectedPatTrigger_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedJets_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi   import *
from PhysicsTools.PatAlgos.slimming.slimmedElectrons_cfi import *

MicroEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_selectedPatPhotons*_*_*',
        #'keep *_selectedPatElectrons*_*_*',
        'keep *_slimmedElectrons_*_*',
        'keep *_selectedPatMuons*_*_*',
        'keep *_selectedPatTaus*_*_*',
        #'keep *_selectedPatJets*_*_*',
        'keep *_slimmedJets_*_*',
        'keep *_patMETs*_*_*',
        ## add extra METs

        'drop *_*_caloTowers_*',
        'drop *_*_pfCandidates_*',
        'drop *_*_genJets_*',

        'keep *_offlineBeamSpot_*_*',
        'keep *_offlineSlimmedPrimaryVertices_*_*',
        'keep patPackedCandidates_packedPFCandidates_*_*',

        #'keep double_*_rho_*', ## need to understand what are the rho's in 70X
        'keep *_selectedPatTrigger_*_PAT',
        'keep *_l1extraParticles_*_HLT',
        'keep *_TriggerResults_*_HLT',

        #'keep *_TriggerResults_*_PAT', # this will be needed for MET filters
    )
)
MicroEventContentMC = cms.PSet(
    outputCommands = cms.untracked.vstring(MicroEventContent.outputCommands)
)
MicroEventContentMC.outputCommands += [
        'keep *_slimmedGenJets_*_*',

        'keep *_prunedGenParticles_*_*',
        'keep LHEEventProduct_source_*_*',
        'keep PileupSummaryInfos_*_*_*',
        'keep GenRunInfoProduct_*_*_*',
        'keep GenFilterInfo_*_*_*',
]
