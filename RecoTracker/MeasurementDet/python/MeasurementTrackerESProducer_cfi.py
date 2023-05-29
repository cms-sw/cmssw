import FWCore.ParameterSet.Config as cms

from RecoTracker.MeasurementDet._MeasurementTrackerESProducer_default_cfi import _MeasurementTrackerESProducer_default

MeasurementTracker = _MeasurementTrackerESProducer_default.clone()

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(MeasurementTracker, 
                             Phase2StripCPE = 'Phase2StripCPE',
                             #StripCPE = '',
                             UseStripModuleQualityDB = False,
                             UseStripAPVFiberQualityDB = False,
                             MaskBadAPVFibers = False,
                             UseStripStripQualityDB = False)

