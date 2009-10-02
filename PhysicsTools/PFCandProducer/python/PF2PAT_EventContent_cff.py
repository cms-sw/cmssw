
import FWCore.ParameterSet.Config as cms


PF2PATEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    # Gen information
    'keep *_genParticles_*_*',
    'keep *_genMetTrue_*_*',
    'keep recoGenJets_*_*_*',
#    'keep *_pfCandToVertexAssociator_*_*',
    # isolated electrons and muons
    'keep *_pfIsolatedElectrons_*_*',
    'keep *_pfIsolatedMuons_*_*',
    'keep *_pfNoJet_*_*',
    'keep recoIsoDepositedmValueMap_*_*_*',
    # jets
    'keep pfRecoPFJets_pfNoTau_*_*',
    # taus 
    'keep *_allLayer0Taus_*_*',
    'keep recoPFTauDiscriminator_*_*_*',
    'keep *_offlinePrimaryVerticesWithBS_*_*',
    # MET
    'keep *_pfMET_*_*'
    )
)

PATEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    # Gen information
#    'keep *_genParticles_*_*',
    'keep *_genMetTrue_*_*',
    'keep recoGenJets_iterativeCone5GenJets_*_*',
    'keep patElectrons_*_*_*',
    'keep patMuons_*_*_*',
    'keep patJets_*_*_*',
    'keep patMET_*_*_*',
    'keep patTaus_*_*_*',
    'keep recoIsoDepositedmValueMap_iso*_*_*'
    )
)

PF2PATStudiesEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoPFJets_*_*_*',
    'keep *_decaysFromZs_*_*',
    'keep recoPFCandidates_*_*_PF2PAT',
    'keep recoPFCandidates_*_*_PAT',    
    'keep recoPFCandidates_particleFlow_*_*',
    'keep recoMuons_*_*_*',
    'keep *_pf*_*_*'
    )
)

