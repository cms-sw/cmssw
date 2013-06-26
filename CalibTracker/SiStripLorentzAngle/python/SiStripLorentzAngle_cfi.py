import FWCore.ParameterSet.Config as cms

read = cms.EDFilter("SiStripLorentzAngle",
    MTCCtrack = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    fileName = cms.string('trackhisto.root')
)


