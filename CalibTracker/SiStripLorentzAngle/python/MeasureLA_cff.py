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

# Calibration found from ensemble of 30 pseudo-datasets at each of 6 points, tanLA = {-0.15,-0.12,-0.09,-0.06,-0.03,0.005}
# Sample was 2/3 of /TkCosmics38T/Summer09-STARTUP31X_V3-v1/GEN-SIM-DIGI-RAW at each tanLA point
# redigitized in peak mode with peak mode noises, CMSSW_3_3_2
LorentzAngleCalibrations_PeakModeCOSMIC = cms.VPSet(
    cms.PSet( Method = METHOD_WIDTH,
              Slopes = cms.vdouble(0.985407,0.986015,0.982955,0.992045,0.988984,0.987343,0.954721,0.95576,0.954863,0.960566,0.959319,0.956621,0.977128,0.976453),
              Offsets = cms.vdouble(0.000363377,0.000347239,-8.73609e-05,0.000633789,0.000785546,0.00063365,0.000723714,0.000720315,0.000644327,0.000814116,0.00164101,0.00123942,0.00180739,0.00171959),
              Pulls = cms.vdouble(1.05217,1.06731,0.97208,1.14943,1.07177,0.988835,0.942119,1.025,1.11031,0.930538,1.08191,1.04613,1.03081,1.17641) ),
    cms.PSet( Method = METHOD_PROB1,
              Slopes = cms.vdouble(0.982061,0.98068,0.976091,0.978934,0.971368,0.971559,0.964595,0.959935,0.964505,0.961342,0.956929,0.972855,0.974695,0.972261),
              Offsets = cms.vdouble(-8.24181e-06,-1.06929e-05,-6.53133e-05,-0.000149074,7.90199e-05,-0.000329831,0.00013365,-0.000472845,-7.92843e-05,-0.000348687,-0.000571311,7.07575e-05,3.87507e-05,-0.000166651),
              Pulls = cms.vdouble(1.07221,0.909626,0.861862,0.990638,1.01422,1.01225,1.05827,1.00192,1.10314,0.916937,0.901404,0.932825,0.922407,0.947059) ),
    cms.PSet( Method = METHOD_AVGV2,
              Slopes = cms.vdouble(0.991046,0.988139,0.992464,0.989286,0.978063,0.970409,0.977013,0.981871,0.976017,0.980569,0.978308,0.977915,0.99055,0.989702),
              Offsets = cms.vdouble(0.000179484,-0.000315592,0.000286534,8.59587e-05,0.000123831,-0.000232847,-0.000386919,9.07647e-05,-0.000112936,-1.02089e-05,0.000157863,0.000102052,0.000240302,-8.68151e-05),
              Pulls = cms.vdouble(0.99978,1.00856,1.0204,1.17001,1.07818,1.01825,1.01474,0.958657,1.00109,1.09488,0.928755,1.00347,1.10553,1.07716) ),
    cms.PSet( Method = METHOD_AVGV3,
              Slopes = cms.vdouble(1.0234,1.01731,1.01839,1.01912,1.03934,1.03615,1.01273,1.00858,1.01181,1.01492,1.01467,1.01318,1.01644,1.00849),
              Offsets = cms.vdouble(0.000702506,0.000216634,3.31415e-05,0.000210135,0.000578267,0.000550148,0.000451231,-1.4985e-05,0.000224097,0.000587055,0.000725274,0.000455932,0.00118975,0.00021313),
              Pulls = cms.vdouble(1.05618,0.912573,1.08878,1.07971,0.957048,0.982786,1.00972,1.01669,0.993294,0.928439,0.980928,0.988378,1.08049,1.16658) ),
    cms.PSet( Method = METHOD_RMSV2,
              Slopes = cms.vdouble(0.974224,0.974741,0.98057,0.987761,0.952585,0.949676,0.955757,0.959137,0.953516,0.954911,0.959072,0.955053,0.963291,0.96448),
              Offsets = cms.vdouble(-0.000242094,-0.000455473,0.000236359,0.000771006,0.000200029,-0.000187993,0.000151929,0.000249159,0.000239323,0.000289015,0.000260618,0.000420144,-0.00024393,-0.000411812),
              Pulls = cms.vdouble(1.30967,1.23259,1.14261,1.24064,1.24455,1.28206,1.37354,1.39888,1.26128,1.30592,1.36302,1.23031,1.40948,1.32819) ),
    cms.PSet( Method = METHOD_RMSV3,
              Slopes = cms.vdouble(1.05309,1.0322,1.02728,1.04035,1.07448,1.0385,1.04547,1.05374,1.04194,1.04169,1.05094,1.04238,1.0321,1.03053),
              Offsets = cms.vdouble(0.00105634,-0.000423326,-0.000977181,-9.60822e-05,-0.000328321,-0.00243066,-0.00163576,-0.000720268,-0.00164131,-0.00167024,-0.000967582,-0.00190743,-0.00107932,-0.00120297),
              Pulls = cms.vdouble(1.24586,1.20579,1.11604,1.22117,1.42447,1.2723,1.31163,1.40946,1.30212,1.4361,1.32736,1.31096,1.35414,1.40196) )
    )

LorentzAngleCalibrations_DeconvolutionModeCOSMIC = cms.VPSet( 
    cms.PSet( Method = METHOD_WIDTH, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_PROB1, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_AVGV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV2, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls),
    cms.PSet( Method = METHOD_RMSV3, Slopes = uncalSlopes , Offsets = uncalOffsets, Pulls = uncalPulls)
    )
