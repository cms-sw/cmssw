import FWCore.ParameterSet.Config as cms

allLayer0CaloTaus = cms.EDFilter("PATCaloTauCleaner",
    tauSource              = cms.InputTag("caloRecoTauProducer"),
    tauDiscriminatorSource = cms.InputTag("caloRecoTauDiscriminationByIsolation"),

    removeOverlaps = cms.PSet(
        ## Flag or discard taus that match with clean electrons
        #electrons = cms.PSet(     
        #    collection = cms.InputTag("allLayer0Electrons")
        #    deltaR     = cms.double(0.3)
        #)
    ),

    markItems    = cms.bool(True),## write the status flags in the output items
    bitsToIgnore = cms.vstring(), ## You can specify some bit names, e.g. "Overflow/User1", "Core/Duplicate", "Isolation/All".
    saveRejected = cms.string(''),## set this to a non empty label to save the list of items which fail
    saveAll      = cms.string(''),## set this to a non empty label to save a list of all items both passing and failing
)


