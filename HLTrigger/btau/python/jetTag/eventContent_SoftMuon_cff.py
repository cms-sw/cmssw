import FWCore.ParameterSet.Config as cms

JetTagSoftMuonHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBSoftmuonHighestEtJets_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltBSoftmuonL25TagInfos_*_*', 
        'keep *_hltBSoftmuonL25BJetTags_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltBSoftmuonL3TagInfos_*_*', 
        'keep *_hltBSoftmuonL3BJetTags_*_*', 
        'keep *_hltBSoftmuonL3BJetTagsByDR_*_*')
)

