import FWCore.ParameterSet.Config as cms

#
# This object is used to make changes for different running scenarios
#
from Configuration.StandardSequences.Eras import eras

SiPixelPhase1DigisAnalyzer = cms.EDAnalyzer("SiPixelPhase1DigisAnalyzer",
	TopFolderName = cms.string('PixelPhase1'),
        src = cms.InputTag("simSiPixelDigis")
)
SiPixelPhase1DigisHarvester = cms.EDAnalyzer("SiPixelPhase1DigisHarvester",
	TopFolderName = cms.string('PixelPhase1'),
        src = cms.InputTag("simSiPixelDigis")
)
