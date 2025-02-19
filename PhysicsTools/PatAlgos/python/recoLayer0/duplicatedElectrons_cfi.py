import FWCore.ParameterSet.Config as cms

# Remove duplicates from the electron list

electronsNoDuplicates = cms.EDFilter("DuplicatedElectronCleaner",
    ## reco electron input source
    electronSource = cms.InputTag("gsfElectrons"), 
)
