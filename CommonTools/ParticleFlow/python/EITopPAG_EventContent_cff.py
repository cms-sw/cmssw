
import FWCore.ParameterSet.Config as cms


EITopPAGEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    # isolated electrons and muons
    'keep *_pfIsolatedElectronsEI_*_*',
    'keep *_pfIsolatedMuonsEI_*_*',
    # jets
    'keep recoPFJets_ak4PFJetsCHSEI_*_*',
    # btags
    'keep *_ak4JetTracksAssociatorAtVertexPFEI_*_*',
    'keep *_impactParameterTagInfosEI_*_*',
    'keep *_secondaryVertexTagInfosEI_*_*',
    'keep *_combinedSecondaryVertexBJetTagsEI_*_*',
    # taus 
    'keep recoPFTaus_pfTausEI_*_*',
    'keep recoPFTauDiscriminator_pfTausDiscrimination*_*_*',
    # MET
    'keep *_pfMetEI_*_*'
    )
)
