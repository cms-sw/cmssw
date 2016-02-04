# The following comments couldn't be translated into the new config version:

#FileInPath filename="CalibCalorimetry/CaloTPG/data/outputLUTtranscoder.dat"

import FWCore.ParameterSet.Config as cms

process = cms.Process("digitize")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi")

process.load("CalibCalorimetry.Configuration.Hcal_FakeConditions_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.HcalTPGCoderULUT = cms.ESProducer("HcalTPGCoderULUT",
    filename = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/RecHit-TPG-calib.dat')
)

process.sw2hw = cms.EDAnalyzer("HcalLuttoDB",
    filePerCrate = cms.untracked.bool(True),
    filePrefix = cms.string('testLUT'),
    targetfirmware = cms.string('1.0.0'),
    creationtag = cms.string('emap_hcal_emulator_test_luts')
)

process.p = cms.Path(process.sw2hw)
process.MessageLogger.cerr.INFO.limit = 1000000

