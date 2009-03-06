import FWCore.ParameterSet.Config as cms

isoTrkCalib = cms.EDAnalyzer("HcalIsoTrkAnalyzer",
    hbheInput = cms.InputTag("IsoProd","IsoTrackHBHERecHitCollection"),
    associationConeSize = cms.double(0.5),
    outputFileName = cms.string('test.root'),
    hoInput = cms.InputTag("IsoProd","IsoTrackHORecHitCollection"),
    allowMissingInputs = cms.bool(False),
    eInput = cms.InputTag("IsoProd","IsoTrackEcalRecHitCollection"),
    trackInput = cms.InputTag("IsoProd","IsoTrackTracksCollection")
)


