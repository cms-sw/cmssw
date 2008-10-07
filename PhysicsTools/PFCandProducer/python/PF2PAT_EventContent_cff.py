
import FWCore.ParameterSet.Config as cms


PF2PATEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    'keep *_pfMET_*_*',
    'keep *_pfCandToVertexAssociator_*_*',
    'keep *_pfPileUp_*_*',
    'keep *_pfElectrons_*_*',
    'keep *_pfMuons_*_*',
    'keep *_pfJets_*_*',
    'keep *_pfTaus_*_*',
    'keep *_pfTopProjection_*_*'
    )
)
