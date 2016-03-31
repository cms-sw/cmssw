import FWCore.ParameterSet.Config as cms

# this might also go into te Common config,as we do not reference it
from DQM.SiPixelPhase1Common.HistogramManager_cfi import *

SiPixelPhase1DigisConf = cms.VPSet(
  DefaultHisto, # ADC
  DefaultHisto, # Ndigis
  DefaultHisto  # hitmaps
)

SiPixelPhase1DigisAnalyzer = cms.EDAnalyzer("SiPixelPhase1DigisAnalyzer",
        src = cms.InputTag("simSiPixelDigis"), #TODO: this should be centralized
	histograms = SiPixelPhase1DigisConf
)
SiPixelPhase1DigisHarvester = cms.EDAnalyzer("SiPixelPhase1DigisHarvester",
	histograms = SiPixelPhase1DigisConf
)
