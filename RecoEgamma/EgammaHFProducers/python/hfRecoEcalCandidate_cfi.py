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
                                     CorrectForPileup= cms.bool(False)
                                     )

#
##                                       hard cut
##                                      intercept2DCut = cms.double(0.92),
##                                      intercept2Dslope = cms.double(0.20),

##                                      medium cut
##                                      intercept2DCut = cms.double(0.875),
##                                      intercept2Dslope = cms.double(0.275),

##                                      loose cut
##                                      intercept2DCut = cms.double(0.815),
##                                      intercept2Dslope = cms.double(0.475),
##                                      era 4=data july 05 3=summer11 MC 2=spring11 MC 1=fall10 MC 0=Data 41
