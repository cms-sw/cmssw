import FWCore.ParameterSet.Config as cms

# HF RecoEcalCandidate Producer
#Values for specific electron cuts and "DataBase" version/vector format below code
hfRecoEcalCandidate = cms.EDProducer("HFRecoEcalCandidateProducer",
                                     e9e25Cut = cms.double(0.94),
                                     hfclusters = cms.InputTag("hfEMClusters"),
                                     VertexCollection = cms.InputTag("offlinePrimaryVertices"),
                                     intercept2DCut = cms.double(0.815),
                                     intercept2DSlope = cms.double(0.475),
                                     Correct = cms.bool(True),
                                     e1e9Cut= cms.vdouble(-1,99),
                                     eCOREe9Cut= cms.vdouble(-1,99),
                                     eSeLCut= cms.vdouble(-1,99),
                                     HFDBversion= cms.int32(1),
                                     HFDBvector=cms.vdouble(
                                                                      #energy corrections
                                                                    
                                                                      1.000,
                                                                      1.000,0.899,0.994,0.958,
                                                                      0.942,0.943,0.960,0.928,
                                                                      0.922,0.896,0.812,1.000,
                                                                      1.000,0.820,0.917,0.952,
                                                                      0.929,0.975,0.984,1.012,
                                                                      0.971,1.016,0.938,1.000,
                                                                      1.000,

                                                                      #start pile up slopes
                                                                      0.0,
                                                                      0.0,-0.0036, -0.0087,-0.0049,
                                                                      -0.0161,-0.0072,-0.0033,-0.0066,
                                                                      -0.0062,-0.0045,-0.0090,0.0,
                                                                      0.0,-0.0056,-0.0024,-0.0064,
                                                                      -0.0063,-0.0078,-0.0079,-0.0075,
                                                                      -0.0074,0.0009,-0.0180,0.0,
                                                                      0.0,
                                                                      #start pile up intercepts
                                                                      1.0,
                                                                      1.0,1.0565,1.0432,1.0714,
                                                                      1.1140,1.0908,1.0576,1.0821,
                                                                      1.0807,1.0885,1.1783,1.0,
                                                                      1.0,1.1570,1.0631,1.0401,
                                                                      1.0803,1.0506,1.0491,1.0235,
                                                                      1.0643,0.9910,1.0489,1.0,
                                                                      1.0)
                                     )

#Electron Cuts
##                                       hard cut
##                                      intercept2DCut = cms.double(0.92),
##                                      intercept2DSlope = cms.double(0.20),

##                                      medium cut
##                                      intercept2DCut = cms.double(0.875),
##                                      intercept2DSlope = cms.double(0.275),

##                                      loose cut
##                                      intercept2DCut = cms.double(0.815),
##                                      intercept2DSlope = cms.double(0.475),


## "DataBase" vector guide:
## version 0: only energy correction, no pileup.
## version 1: energy correction AND pileup
##for parts of the vector that depend on ieta, they follow this pattern.
## -41,
## -40,-39,-38,-37,
## -36,-35,-34,-33,
## -32,-31,-30,-29,
##  29, 30, 31, 32,
##  33, 34, 35, 36,
##  37, 38, 39, 40,
##  41


