import FWCore.ParameterSet.Config as cms

dtTTrigWriter = cms.EDAnalyzer("DTTTrigWriter",
    # Switch on/off the verbosity
    debug = cms.untracked.bool(False),
    # Name of the input ROOT file which contains the time boxes
    rootFileName = cms.untracked.string('DTTimeBoxes.root'),
    # the kfactor to be uploaded in the ttrig DB
    kFactor = cms.untracked.double(-0.7)
)
