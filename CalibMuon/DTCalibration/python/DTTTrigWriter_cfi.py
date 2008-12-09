import FWCore.ParameterSet.Config as cms

ttrigwriter = cms.EDFilter("DTTTrigWriter",
                           # Switch on/off the verbosity
                           debug = cms.untracked.bool(True),
                           # Name of the input ROOT file which contains the time boxes
                           rootFileName = cms.untracked.string('DTTimeBoxes_TEMPLATE.root'),
                           # the kfactor to be uploaded in the ttrig DB
                           kFactor = cms.untracked.double(-0.7)
                           )
