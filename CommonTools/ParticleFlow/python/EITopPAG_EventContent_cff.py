
import FWCore.ParameterSet.Config as cms


EITopPAGEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    # isolated electrons and muons
    'keep *_pfIsolatedElectronsEI*_*_*',
    'keep *_pfIsolatedMuonsEI*_*_*',
    # jets
    'keep recoPFJets_pfJets_*_*',
    # taus 
    'keep recoPFTaus_pfTaus_*_*',
    'keep recoPFTauDiscriminator_*_*_*',
    'keep *_*fflinePrimaryVertices*_*_*',
    # MET
    'keep *_pfMET*_*_*'
    )
)
