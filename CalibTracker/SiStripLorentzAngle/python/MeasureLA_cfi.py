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
    InFileLocation = cms.string('/calibrationTree/tree'),
    SiStripDetInfo = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
    #MaxEvents = cms.untracked.uint32(0),
    Report = cms.VPSet( LA_Report( METHOD_SQRTVAR, byLayer, "sqrtvarByLayer"),
                        LA_Report( METHOD_RATIO,   byLayer, "ratioByLayer"),
                        LA_Report( METHOD_WIDTH,   byLayer, "widthByLayer")
                        ),
    MeasurementPreference = cms.VPSet(  LA_Measurement( METHOD_SQRTVAR, byModule, 1000, 3.0),
                                        LA_Measurement( METHOD_WIDTH,   byModule, 1000, 3.0),
                                        LA_Measurement( METHOD_SQRTVAR, byLayer,  1000, 3.5)
                                        ),
    Calibrations = LorentzAngleCalibrations_PeakMode
    )
