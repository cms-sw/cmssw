import FWCore.ParameterSet.Config as cms

# HF RecoEcalCandidate Producer
hfRecoEcalCandidate = cms.EDProducer("HFRecoEcalCandidateProducer",
                                     e9e25Cut = cms.double(0.94),
                                     hfclusters = cms.InputTag("hfEMClusters"),
                                     intercept2DCut = cms.double(0.815),
                                     intercept2DSlope = cms.double(0.475),
                                     Correct = cms.bool(True),
                                     e1e9Cut= cms.vdouble(-1,99),
                                     eCOREe9Cut= cms.vdouble(-1,99),
                                     eSeLCut= cms.vdouble(-1,99),
                                     era= cms.int32(4),
                                     CorrectForPileup= cms.untracked.bool(False),
                                     PileupSlopes=cms.untracked.vdouble(-0.0036,
                                                                       -0.0087,
                                                                       -0.0049,
                                                                       -0.0161,
                                                                       -0.0072,
                                                                       -0.0033,
                                                                       -0.0066,
                                                                       -0.0062,
                                                                       -0.0045,
                                                                       -0.0090,
                                                                       -0.0056,
                                                                       -0.0024,
                                                                       -0.0064,
                                                                       -0.0063,
                                                                       -0.0078,
                                                                       -0.0079,
                                                                       -0.0075,
                                                                       -0.0074,
                                                                       0.0009,
                                                                       -0.0180),
                                     PileupIntercepts=cms.untracked.vdouble(1.0565,
                                                                           1.0432,
                                                                           1.0714,
                                                                           1.1140,
                                                                           1.0908,
                                                                           1.0576,
                                                                           1.0821,
                                                                           1.0807,
                                                                           1.0885,
                                                                           1.1783,
                                                                           1.1570,
                                                                           1.0631,
                                                                           1.0401,
                                                                           1.0803,
                                                                           1.0506,
                                                                           1.0491,
                                                                           1.0235,
                                                                           1.0643,
                                                                           0.9910,
                                                                           1.0489)
                                     )

#
##                                       hard cut
##                                      intercept2DCut = cms.double(0.92),
##                                      intercept2DSlope = cms.double(0.20),

##                                      medium cut
##                                      intercept2DCut = cms.double(0.875),
##                                      intercept2DSlope = cms.double(0.275),

##                                      loose cut
##                                      intercept2DCut = cms.double(0.815),
##                                      intercept2DSlope = cms.double(0.475),
##                                      era 4=data july 05 3=summer11 MC 2=spring11 MC 1=fall10 MC 0=Data 41
