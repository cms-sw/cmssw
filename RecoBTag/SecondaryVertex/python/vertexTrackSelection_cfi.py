import FWCore.ParameterSet.Config as cms

vertexTrackSelectionBlock = cms.PSet(
	trackSelection = cms.PSet(
		totalHitsMin = cms.uint32(8),
		jetDeltaRMax = cms.double(0.3),
		qualityClass = cms.string('highPurity'),
		pixelHitsMin = cms.uint32(2),
		sip3dSigMin = cms.double(-99999.9),
		sip3dSigMax = cms.double(99999.9),
		sip2dValMax = cms.double(99999.9),
		ptMin = cms.double(1.0),
		sip2dSigMax = cms.double(99999.9),
		sip2dSigMin = cms.double(-99999.9),
		sip3dValMax = cms.double(99999.9),
		sip3dValMin = cms.double(-99999.9),
		sip2dValMin = cms.double(-99999.9),
		normChi2Max = cms.double(99999.9)
	)
)
