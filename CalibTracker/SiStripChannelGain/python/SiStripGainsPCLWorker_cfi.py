import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiStripGainsPCLWorker = DQMEDAnalyzer( 
    "SiStripGainsPCLWorker",
    minTrackMomentum    = cms.untracked.double(2),
    maxNrStrips         = cms.untracked.uint32(8),
    Validation          = cms.untracked.bool(False),
    OldGainRemoving     = cms.untracked.bool(False),
    FirstSetOfConstants = cms.untracked.bool(True),
    UseCalibration      = cms.untracked.bool(False),
    DQMdir              = cms.untracked.string('AlCaReco/SiStripGains'),
    calibrationMode     = cms.untracked.string('StdBunch'),
    ChargeHisto         = cms.untracked.vstring('TIB','TIB_layer_1','TOB','TOB_layer_1','TIDminus','TIDplus','TECminus','TECplus','TEC_thin','TEC_thick'),
    tracks=cms.InputTag("generalTracks",""),
    )
