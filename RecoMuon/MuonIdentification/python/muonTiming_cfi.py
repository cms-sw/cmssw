import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentification.MuonTimingFiller_cfi import *

muontiming = cms.EDProducer('MuonTimingProducer',
  TimingFillerBlock,
  MuonCollection = cms.InputTag("muons1stStep"),
)
