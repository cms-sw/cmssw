import FWCore.ParameterSet.Config as cms

from Calibration.LumiAlCaRecoProducers.AlcaPCCProduer_cfi import*
alcaPCC = cms.Sequence( alcaPCCProducer )
