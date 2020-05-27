import FWCore.ParameterSet.Config as cms

from RecoTracker.MeasurementDet._MeasurementTrackerESProducer_default_cfi import _MeasurementTrackerESProducer_default

MeasurementTracker = _MeasurementTrackerESProducer_default.clone()

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(MeasurementTracker, 
                             Phase2StripCPE = cms.string('Phase2StripCPE'),
                             #StripCPE = cms.string(''),
                             UseStripModuleQualityDB = cms.bool(False),
                             UseStripAPVFiberQualityDB = cms.bool(False),
                             MaskBadAPVFibers = cms.bool(False),
                             UseStripStripQualityDB = cms.bool(False),
                             )

from Configuration.Eras.Modifier_phase2_brickedPixels_cff import phase2_brickedPixels
phase2_brickedPixels.toModify(MeasurementTracker, PixelCPE = cms.string('PixelCPEGenericForBricked'))

