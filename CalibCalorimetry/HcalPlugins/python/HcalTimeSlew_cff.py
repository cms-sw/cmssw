import FWCore.ParameterSet.Config as cms

# Parameters needed for the TimeSlew models for M2 and M3

timeSlewParametersM2 = cms.VPSet(
    cms.PSet(bias = cms.string("Slow"),   tzero = cms.double(23.960177), slope = cms.double(-3.178648), tmax = cms.double(16.00)),
    cms.PSet(bias = cms.string("Medium"), tzero = cms.double(13.307784), slope = cms.double(-1.556668), tmax = cms.double(10.00)),
    cms.PSet(bias = cms.string("Fast"),   tzero = cms.double(9.109694),  slope = cms.double(-1.075824), tmax = cms.double(6.25))
)

timeSlewParametersM3 = cms.VPSet(
    cms.PSet(cap = cms.double(6.0), tspar0 = cms.double(15.5),    tspar1 = cms.double(-3.2),     tspar2 = cms.double(32.0), tspar0_siPM = cms.double(0.0), tspar1_siPM = cms.double(0.0), tspar2_siPM = cms.double(0.0)),
    cms.PSet(cap = cms.double(6.0), tspar0 = cms.double(12.2999), tspar1 = cms.double(-2.19142), tspar2 = cms.double(0.0),  tspar0_siPM = cms.double(0.0), tspar1_siPM = cms.double(0.0), tspar2_siPM = cms.double(0.0))
)

#tzero        = {23.960177, 13.307784, 9.109694};
#slope        = {-3.178648,  -1.556668, -1.075824 };
#tmax         = {16.00, 10.00, 6.25 };
#cap          = 6.0;
#tspar0       = {15.5, 12.2999};
#tspar1       = {-3.2,-2.19142};
#tspar2       = {32, 0};
#tspar0_siPM  = {0., 0.}; // 0ns delay for MC and DATA, recheck later for data
#tspar1_siPM  = {0, 0};
#tspar2_siPM  = {0, 0};
