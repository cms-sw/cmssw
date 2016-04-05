import FWCore.ParameterSet.Config as cms

from Alignment.MillePedeAlignmentAlgorithm.MillePedeFileReader_cfi import *

SiPixelAliDQMModule = cms.EDAnalyzer("MillePedeDQMModule",
                                     MillePedeFileReader = cms.PSet(MillePedeFileReader)
    )
