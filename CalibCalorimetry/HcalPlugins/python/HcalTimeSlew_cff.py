import FWCore.ParameterSet.Config as cms

HcalTimeSlewEP = cms.ESSource("HcalTimeSlewEP",
    # for method2
    timeSlewParametersM2 = cms.VPSet(
        cms.PSet(bias = cms.string("Slow"),    tzero = cms.double(23.960177), slope = cms.double(-3.178648), tmax = cms.double(16.00)),
        cms.PSet(bias = cms.string("Medium"),  tzero = cms.double(13.307784), slope = cms.double(-1.556668), tmax = cms.double(10.00)),
        cms.PSet(bias = cms.string("Fast"),    tzero = cms.double(9.109694),  slope = cms.double(-1.075824), tmax = cms.double(6.25)),
        cms.PSet(bias = cms.string("HBHE2018"),tzero = cms.double(11.977461), slope = cms.double(-1.5610227),tmax = cms.double(10.00))
    ),
    # for method3                          
    timeSlewParametersM3 = cms.VPSet(
        cms.PSet(cap = cms.double(6.0), tspar0 = cms.double(15.5),    tspar1 = cms.double(-3.2),     tspar2 = cms.double(32.0), tspar0_siPM = cms.double(0.0), tspar1_siPM = cms.double(0.0), tspar2_siPM = cms.double(0.0)),
        cms.PSet(cap = cms.double(6.0), tspar0 = cms.double(12.2999), tspar1 = cms.double(-2.19142), tspar2 = cms.double(0.0),  tspar0_siPM = cms.double(0.0), tspar1_siPM = cms.double(0.0), tspar2_siPM = cms.double(0.0))
    )
)



