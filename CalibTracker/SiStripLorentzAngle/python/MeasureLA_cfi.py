import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripLorentzAngle.MeasureLA_cff import *

SiStripLorentzAngleRcdSource = cms.ESSource( "EmptyESSource",
                                             recordName = cms.string( "SiStripLorentzAngleRcd" ),
                                             iovIsRunNotTime = cms.bool( True ),
                                             firstValid = cms.vuint32( 1 )
                                             )

MeasurementLA = cms.ESProducer(
    "sistrip::MeasureLA",
    InputFiles = cms.vstring([]),
    InFileLocaltion = cms.string('/calibrationTree/tree'),
    #MaxEvents = cms.untracked.uint32(0),
    Report = cms.VPSet( LA_Report( METHOD_SQRTVAR, byLayer),
                        LA_Report( METHOD_RATIO,   byLayer),
                        LA_Report( METHOD_WIDTH,   byLayer)
                        ),
    MeasurementPreference = cms.VPSet(  LA_Measurement( METHOD_SQRTVAR, byModule, 1000, 3.0),
                                        LA_Measurement( METHOD_WIDTH,   byModule, 1000, 3.0),
                                        LA_Measurement( METHOD_SQRTVAR, byLayer,  1000, 3.5)
                                        ),
    Calibrations = LorentzAngleCalibrations_PeakMode
    )
