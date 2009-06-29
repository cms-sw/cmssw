import FWCore.ParameterSet.Config as cms



lightChHiggsToTauNuFilter = cms.EDFilter("LightChHiggsToTauNuSkim",


    # Collection to be accessed
    jetsTag       = cms.InputTag("sisCone5CaloJets"),
    muonsTag      = cms.InputTag("globalMuons"),
    electronsTag  = cms.InputTag("pixelMatchGsfElectrons"),


    # Lepton Thresholds
    leptonPtMin   = cms.double(15),
    leptonEtaMin  = cms.double(-2.4),
    leptonEtaMax  = cms.double(2.4),

    # jet thresolds
    minNumbOfJets = cms.int32(2),
    jetPtMin      = cms.double(15),
    jetEtaMin     = cms.double(-2.4),
    jetEtaMax     = cms.double(2.4)
       
)

