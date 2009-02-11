import FWCore.ParameterSet.Config as cms

allLayer0METs = cms.EDFilter("PATCaloMETCleaner",
    ## Input MET from AOD
    metSource = cms.InputTag('corMetType1Icone5Muons'), ## met corrected for jets and for muons
    #metSource = cms.InputTag('met'),                     ## NO MET CORRECTIONS

    markItems = cms.bool(True),    ## write the status flags in the output items
    bitsToIgnore = cms.vstring(),  ## You can specify some bit names, e.g. "Overflow/User1", "Core/Duplicate", "Isolation/All".
    saveRejected = cms.string(''), ## set this to a non empty label to save the list of items which fail
    saveAll = cms.string(''),      ## set this to a non empty label to save a list of all items both passing and failing
)


