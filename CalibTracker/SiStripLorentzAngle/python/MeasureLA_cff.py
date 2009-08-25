import FWCore.ParameterSet.Config as cms

byLayer = cms.bool(False)
byModule = cms.bool(True)
METHOD_WIDTH = cms.int32(1)
METHOD_RATIO = cms.int32(2)
METHOD_SQRTVAR = cms.int32(4)

def LA_Report(method, granularity, filename) :
    return cms.PSet( Method = method,
                     ByModule = granularity,
                     ReportName = cms.string(filename) )

def LA_Measurement(method, granularity, minimumEntries, maxChi2ndof) :
    return cms.PSet( Method = method,
                     ByModule = granularity,
                     MinEntries = cms.uint32(minimumEntries),
                     MaxChi2ndof = cms.double(maxChi2ndof) )

def LA_Calibration(method, pitch, slope, offset, pull) :
    return cms.PSet( Method = method,
                     Pitch = cms.int32(pitch),
                     Slope = cms.double(slope),
                     Offset = cms.double(offset),
                     ErrorScaling = cms.double(pull) )


LorentzAngleCalibrations_PeakMode = cms.VPSet(
    LA_Calibration( METHOD_SQRTVAR, 183, 1, 0, 1),
    LA_Calibration( METHOD_SQRTVAR, 122, 1, 0, 1),
    LA_Calibration( METHOD_SQRTVAR, 120, 1, 0, 1),
    LA_Calibration( METHOD_SQRTVAR,  80, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,   183, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,   122, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,   120, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,    80, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,   183, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,   122, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,   120, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,    80, 1, 0, 1)
    )

LorentzAngleCalibrations_DeconvolutionMode = cms.VPSet(
    LA_Calibration( METHOD_SQRTVAR, 183, 1, 0, 1),
    LA_Calibration( METHOD_SQRTVAR, 122, 1, 0, 1),
    LA_Calibration( METHOD_SQRTVAR, 120, 1, 0, 1),
    LA_Calibration( METHOD_SQRTVAR,  80, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,   183, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,   122, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,   120, 1, 0, 1),
    LA_Calibration( METHOD_RATIO,    80, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,   183, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,   122, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,   120, 1, 0, 1),
    LA_Calibration( METHOD_WIDTH,    80, 1, 0, 1)
    )
