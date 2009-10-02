import FWCore.ParameterSet.Config as cms


pfMET = cms.EDProducer("PFMET",
    PFCandidates = cms.InputTag("particleFlow"),
    # HF calibration factor 1/0.7=1.429 (in 31X applied by PFProducer)
    hfCalibFactor =  cms.double(1.0),
    verbose = cms.untracked.bool(False)
)

