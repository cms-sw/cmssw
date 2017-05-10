#
# Configuration options for HcalNoiseAnalyzer
#
import FWCore.ParameterSet.Config as cms

hcalNoiseAnalyzer = cms.EDAnalyzer(
    "HcalNoiseAnalyzer",
    FillHBHE = cms.untracked.bool(True),
    FillHF = cms.untracked.bool(False),
    FillHO = cms.untracked.bool(False),
    TotalChargeThreshold = cms.untracked.double(-1.0e100),
    HBHERecHits = cms.untracked.string("hbhereco")
)
