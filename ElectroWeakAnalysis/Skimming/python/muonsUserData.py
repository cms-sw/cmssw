import FWCore.ParameterSet.Config as cms


userDataMuons = cms.EDProducer(
    "muonUserData",
    src = cms.InputTag("selectedPatMuonsTriggerMatch"),
    ptThreshold = cms.double("1.5"),
    etEcalThreshold = cms.double("0.2"),
    etHcalThreshold = cms.double("0.5"),
    deltaRVetoTrk = cms.double("0.015"),
    deltaRTrk = cms.double("0.3"),
    deltaREcal = cms.double("0.25"),
    deltaRHcal = cms.double("0.25"),
    alpha = cms.double("0."),
    beta = cms.double("-0.75"),
    relativeIsolation = cms.bool(False)
    
    )

#process.out = cms.OutputModule(
#    "PoolOutputModule",
#    fileName = cms.untracked.string('muonUserData.root'),
#    outputCommands = cms.untracked.vstring(
#      "keep *"
#      #"keep *_muons_*_*"  
#    
 #   )
#)


#process.p=cms.Path(process.muons+process.out)
