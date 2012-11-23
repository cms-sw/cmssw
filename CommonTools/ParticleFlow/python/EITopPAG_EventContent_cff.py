
import FWCore.ParameterSet.Config as cms


EITopPAGEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    # isolated electrons and muons
    'keep *_pfIsolatedElectronsEI_*_*',
    'keep *_pfIsolatedMuonsEI_*_*',
    # jets
    'keep recoPFJets_pfJets_*_*',
    # taus 
    'keep recoPFTaus_pfTaus_*_*',
    'keep recoPFTauDiscriminator_pfTausDiscrimination*_*_*',
    # MET
    'keep *_pfMetEI_*_*'
    )
)
