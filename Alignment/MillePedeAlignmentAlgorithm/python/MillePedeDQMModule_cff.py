import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

import  Alignment.MillePedeAlignmentAlgorithm.MillePedeFileReader_cfi as MillePedeFileReader_cfi

SiPixelAliDQMModule = DQMEDHarvester("MillePedeDQMModule",
                                     alignmentTokenSrc = cms.InputTag("SiPixelAliPedeAlignmentProducer"),
                                     MillePedeFileReader = cms.PSet(MillePedeFileReader_cfi.MillePedeFileReader.clone())
                                     )
