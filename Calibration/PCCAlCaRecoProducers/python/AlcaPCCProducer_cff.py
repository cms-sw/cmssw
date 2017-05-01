import FWCore.ParameterSet.Config as cms

from Calibration.PCCAlCaRecoProducers.AlcaPCCProduer_cfi import*
alcaPCC = cms.Sequence( alcaPCCProducer )
