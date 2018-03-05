import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.AlcaBeamSpotProducerHP_cfi import alcaBeamSpotProducerHP
alcaBeamSpotHP = cms.Sequence( alcaBeamSpotProducerHP )
