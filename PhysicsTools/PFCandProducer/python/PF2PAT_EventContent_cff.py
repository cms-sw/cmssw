
import FWCore.ParameterSet.Config as cms


PF2PATEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    # Gen information
    'keep *_genParticles_*_*',
    'keep *_genMetTrue_*_*',
    'keep recoGenJets_*_*_*',
#    'keep *_pfCandToVertexAssociator_*_*',
    # isolated electrons and muons
    'keep *_isolatedElectrons_*_*',
    'keep *_isolatedMuons_*_*',
    'keep *_noJet_*_*',
    'keep recoIsoDepositedmValueMap_*_*_*',
    # jets
    'keep recoPFJets_noTau_*_*',
    # taus 
    'keep *_allLayer0Taus_*_*',
    'keep recoPFTauDiscriminator_*_*_*',
    'keep *_offlinePrimaryVerticesWithBS_*_*',
    # MET
    'keep *_pfMET_*_*'
    )
)

PF2PATStudiesEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep recoPFJets_*_*_*',
    'keep *_decaysFromZs_*_*',
    'keep recoPFCandidates_*_*_PF2PAT',
    'keep recoPFCandidates_*_*_PAT',    
    'keep recoPFCandidates_particleFlow_*_*'
    )
)

