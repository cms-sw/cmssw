import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentification.DTTimingExtractor_cfi import *

muontiming = cms.EDProducer('MuonTimingProducer',
  DTTimingExtractorBlock,
  MuonCollection = cms.InputTag("muons"),
)
