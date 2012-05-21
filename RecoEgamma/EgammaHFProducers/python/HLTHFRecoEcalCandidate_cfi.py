import FWCore.ParameterSet.Config as cms

# HF RecoEcalCandidate Producer
#Values for specific electron cuts and "DataBase" version/vector format below code
HLTHFRecoEcalCandidate = cms.EDProducer("HLTHFRecoEcalCandidateProducer",
                                        e9e25Cut = cms.double(0.94),
                                        hfclusters = cms.InputTag("hfEMClusters"),
                                        Correct = cms.bool(True),
                                        intercept2DCut = cms.double(0.7),
                                        intercept2DSlope = cms.double(0.475),
                                        e1e9Cut= cms.vdouble(-1,99),
                                        eCOREe9Cut= cms.vdouble(-1,99),
                                        eSeLCut= cms.vdouble(-1,99)
                                        
                                        )
                                       

#Electron Cuts for HLT
##                                      tight cut
##                                      intercept2DCut = cms.double(0.7),
##                                      intercept2DSlope = cms.double(0.475),







