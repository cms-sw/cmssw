import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.packedPFCandidates_cfi import *
from PhysicsTools.PatAlgos.slimming.lostTracks_cfi import *
from PhysicsTools.PatAlgos.slimming.offlineSlimmedPrimaryVertices_cfi import *
from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from PhysicsTools.PatAlgos.slimming.selectedPatTrigger_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedJets_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi   import *
from PhysicsTools.PatAlgos.slimming.slimmedElectrons_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedMuons_cfi     import *
from PhysicsTools.PatAlgos.slimming.slimmedPhotons_cfi   import *
from PhysicsTools.PatAlgos.slimming.slimmedTaus_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedSecondaryVertices_cfi      import *
from PhysicsTools.PatAlgos.slimming.slimmedMETs_cfi      import *
from PhysicsTools.PatAlgos.slimming.metFilterPaths_cff   import *
from RecoEgamma.EgammaPhotonProducers.reducedEgamma_cfi  import *

MicroEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_slimmedPhotons*_*_*',
        'keep *_slimmedElectrons_*_*',
        'keep *_slimmedMuons*_*_*',
        'keep *_slimmedTaus*_*_*',
        'keep *_slimmedJets*_*_*',
        'keep *_slimmedMETs*_*_*',
        'keep *_slimmedSecondaryVertices*_*_*',
        ## add extra METs

        'keep recoPhotonCores_reducedEgamma_*_*',
        'keep recoGsfElectronCores_reducedEgamma_*_*',
        'keep recoConversions_reducedEgamma_*_*',
        'keep recoSuperClusters_reducedEgamma_*_*',
        'keep recoCaloClusters_reducedEgamma_*_*',
        'keep EcalRecHitsSorted_reducedEgamma_*_*',
        

        'drop *_*_caloTowers_*',
        'drop *_*_pfCandidates_*',
        'drop *_*_genJets_*',

        'keep *_offlineBeamSpot_*_*',
        'keep *_offlineSlimmedPrimaryVertices_*_*',
        'keep patPackedCandidates_packedPFCandidates_*_*',

        'keep double_fixedGridRho*__*', 
        'keep double_ak5*_rho_*', 
        'keep doubles_ak5*_rhos_*', 

        'keep *_selectedPatTrigger_*_PAT',
        'keep patPackedTriggerPrescales_patTrigger__PAT',
        'keep *_l1extraParticles_*_HLT',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_HLT',
        'keep *_TriggerResults_*_HLT',
        'keep *_TriggerResults_*_PAT', # for MET filters
	'keep patPackedCandidates_lostTracks_*_PAT',
	'keep HcalNoiseSummary_hcalnoise__*'
    )
)
MicroEventContentMC = cms.PSet(
    outputCommands = cms.untracked.vstring(MicroEventContent.outputCommands)
)
MicroEventContentMC.outputCommands += [
        'keep *_slimmedGenJets_*_*',
        'keep *_packedGenParticles_*_*',
        'keep recoGenParticles_prunedGenParticles_*_*',
        'keep LHEEventProduct_*_*_*',
        'keep PileupSummaryInfos_*_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        # RUN
        'keep LHERunInfoProduct_*_*_*',
        'keep GenRunInfoProduct_*_*_*',
        'keep L1GtTriggerMenuLite_l1GtTriggerMenuLite__*',
]
