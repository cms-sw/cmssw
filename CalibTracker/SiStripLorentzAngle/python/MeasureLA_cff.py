import FWCore.ParameterSet.Config as cms

byLayer = cms.int32(0)
byModule = cms.int32(1)
byModuleSummary = cms.int32(2)

METHOD_WIDTH =  cms.int32(  1 )
METHOD_PROB1 =  cms.int32(  2 )
METHOD_AVGV2 =  cms.int32(  4 )
METHOD_AVGV3 =  cms.int32(  8 )
METHOD_RMSV2 =  cms.int32( 16 )
METHOD_RMSV3 =  cms.int32( 32 )

def LA_Report(method, granularity, filename) :
    return cms.PSet( Method = method,
                     Granularity = granularity,
                     ReportName = cms.string(filename) )

def LA_Measurement(method, granularity, minimumEntries, maxChi2ndof) :
    return cms.PSet( Method = method,
                     Granularity = granularity,
                     MinEntries = cms.uint32(minimumEntries),
                     MaxChi2ndof = cms.double(maxChi2ndof) )

uncalSlopes  = cms.vdouble(1,1,1,1,1,1,1,1,1,1,1,1,1,1)
uncalOffsets = cms.vdouble(0,0,0,0,0,0,0,0,0,0,0,0,0,0)
uncalPulls   = cms.vdouble(1,1,1,1,1,1,1,1,1,1,1,1,1,1)

LorentzAngleCalibrations_Uncalibrated = cms.VPSet(
    cms.PSet( Method = METHOD_WIDTH, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_PROB1, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls)
    )

LorentzAngleCalibrations_PeakModeBEAM = cms.VPSet(
    cms.PSet( Method = METHOD_WIDTH, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_PROB1, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls)
    )

LorentzAngleCalibrations_DeconvolutionModeBEAM = cms.VPSet(
    cms.PSet( Method = METHOD_WIDTH, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_PROB1, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls)
    )

LorentzAngleCalibrations_PeakModeCOSMIC = cms.VPSet(
    cms.PSet( Method = METHOD_WIDTH, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_PROB1, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls)
    )

LorentzAngleCalibrations_DeconvolutionModeCOSMIC = cms.VPSet( 
    cms.PSet( Method = METHOD_WIDTH, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_PROB1, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls)
    )
