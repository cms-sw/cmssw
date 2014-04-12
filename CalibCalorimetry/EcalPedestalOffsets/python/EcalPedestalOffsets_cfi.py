import FWCore.ParameterSet.Config as cms

pedTest = cms.EDAnalyzer("EcalPedOffset",
    dbUserName = cms.untracked.string('foo'),
    xmlFile = cms.string('provaSuperWide'),
    EEdigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    run = cms.int32(100),
    dbHostPort = cms.untracked.int32(0),
    EBdigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    maxChi2OverNDF = cms.untracked.double(5.25),
    createMonIOV = cms.untracked.bool(False),
    bestPed = cms.untracked.int32(200),
    plotting = cms.string('details'),
    minSlopeAllowed = cms.untracked.double(-18.0),
    maxSlopeAllowed = cms.untracked.double(-29.0),
    location = cms.untracked.string('foo'),
    DACmin = cms.untracked.int32(40),
    dbPassword = cms.untracked.string('foo'),
    DACmax = cms.untracked.int32(90),
    headerCollection = cms.InputTag("ecalEBunpacker"),
    dbName = cms.untracked.string('0'),
    RMSmax = cms.untracked.double(20.0),
    # DB info
    dbHostName = cms.untracked.string('0')
)


