import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripLorentzAngle.MeasureLA_cff import *

SiStripLorentzAngleRcdSource = cms.ESSource( "EmptyESSource",
                                             recordName = cms.string( "SiStripLorentzAngleRcd" ),
                                             iovIsRunNotTime = cms.bool( True ),
                                             firstValid = cms.vuint32( 1 )
                                             )

MeasureLA = cms.ESProducer(
    "sistrip::MeasureLA",
    InputFiles = cms.vstring([]),
    InFileLocation = cms.string('/calibrationTree/tree'),
    SiStripDetInfo = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
    #MaxEvents = cms.untracked.uint32(100000),
    Reports = cms.VPSet( LA_Report( METHOD_WIDTH,   byLayer, "width_ByLayer"),
                         LA_Report( METHOD_PROB1,   byLayer, "prob1_ByLayer"),
                         LA_Report( METHOD_AVGV2,   byLayer, "avgv2_ByLayer"),
                         LA_Report( METHOD_AVGV3,   byLayer, "avgv3_ByLayer"),
                         LA_Report( METHOD_RMSV2,   byLayer, "rmsv2_ByLayer"),
                         LA_Report( METHOD_RMSV3,   byLayer, "rmsv3_ByLayer")
                         ),
    MeasurementPreferences = cms.VPSet( #LA_Measurement( METHOD_PROB1, byModule, 1000, 3.0),
                                        #LA_Measurement( METHOD_WIDTH, byModule, 1000, 3.0),
                                        #LA_Measurement( METHOD_AVGV2, byLayer,  1000, 3.5)
                                        ),
    Calibrations = LorentzAngleCalibrations_DeconvolutionModeBEAM
    )
