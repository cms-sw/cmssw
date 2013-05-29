import FWCore.ParameterSet.Config as cms

from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProd_cff import *
# adapt the names for the HI AlcaRecos
ALCARECOTkAlMinBiasFilterForBS.HLTPaths = ['pathALCARECOTkAlMinBiasHI']
alcaBeamSpotProducer.BeamFitter.TrackCollection = 'ALCARECOTkAlMinBiasHI'
alcaBeamSpotProducer.BeamFitter.TrackQuality = cms.untracked.vstring('highPurity')
alcaBeamSpotProducer.PVFitter.VertexCollection = 'hiSelectedVertex'
