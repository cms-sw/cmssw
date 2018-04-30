import FWCore.ParameterSet.Config as cms

pfCandidatesBadHadRecalibrated = cms.EDProducer("PFCandidateRecalibrator",
                                                pfcandidates = cms.InputTag("particleFlow"),
                                                shortFibreThr = cms.double(1.4),  #taking it from particleFlowClusterHF_cfi.py
                                                longFibreThr = cms.double(1.4)    #taking it from particleFlowClusterHF_cfi.py
                                                )
