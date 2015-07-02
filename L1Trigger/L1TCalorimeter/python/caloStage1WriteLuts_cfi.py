import FWCore.ParameterSet.Config as cms

## from L1Trigger.L1TCalorimeter.caloStage1Params_cfi import *

writeLuts = cms.EDAnalyzer("L1TCaloStage1LutWriter",
    writeIsoTauLut = cms.untracked.bool(False),
    isoTauLutName = cms.untracked.string("isoTauLut.txt"),
    conditionsLabel = cms.string('')
                                )


