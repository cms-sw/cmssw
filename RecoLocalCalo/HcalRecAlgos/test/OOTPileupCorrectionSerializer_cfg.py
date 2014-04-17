import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HcalRecAlgos.OOTPileupCorrectionSerializer_cfi import *

process = cms.Process('OOTPileupCorrectionSerializer')

process.source = cms.Source('EmptySource')
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(0))

process.filewriter = ootPileupCorrectionSerializer
process.filewriter.outputFile = cms.string("testOOTPileupCorrection.gssa")

process.p = cms.Path(process.filewriter)
