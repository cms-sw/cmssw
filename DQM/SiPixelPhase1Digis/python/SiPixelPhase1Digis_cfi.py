import FWCore.ParameterSet.Config as cms

#
# This object is used to make changes for different running scenarios
#
from Configuration.StandardSequences.Eras import eras

SiPixelPhase1Digis = cms.EDAnalyzer("SiPixelPhase1Digis",
	TopFolderName = cms.string('PixelPhase1'),
        src = cms.InputTag("simSiPixelDigis")
)
