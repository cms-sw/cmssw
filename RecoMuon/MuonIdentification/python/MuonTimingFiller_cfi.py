import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentification.DTTimingExtractor_cfi import *
from RecoMuon.MuonIdentification.CSCTimingExtractor_cfi import *

TimingFillerBlock = cms.PSet(
  TimingFillerParameters = cms.PSet(
    DTTimingExtractorBlock,
    CSCTimingExtractorBlock
  )
)


