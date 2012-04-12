import FWCore.ParameterSet.Config as cms

# HF RecoEcalCandidate Producer
hfRecoEcalCandidate = cms.EDProducer("HFRecoEcalCandidateProducer",
                                     e9e25Cut = cms.double(0.94),
                                     hfclusters = cms.InputTag("hfEMClusters"),
                                     intercept2DCut = cms.double(0.3),
                                     Correct = cms.bool(True),
                                     e1e9Cut= cms.vdouble(-1,99),
                                     eCOREe9Cut= cms.vdouble(-1,99),
                                     eSeLCut= cms.vdouble(-1,99)
                                     )



