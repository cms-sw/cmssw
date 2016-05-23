import FWCore.ParameterSet.Config as cms

import  Alignment.MillePedeAlignmentAlgorithm.MillePedeFileReader_cfi as MillePedeFileReader_cfi

SiPixelAliDQMModule = cms.EDAnalyzer("MillePedeDQMModule",
                                     MillePedeFileReader = cms.PSet(MillePedeFileReader_cfi.MillePedeFileReader.clone())
    )
