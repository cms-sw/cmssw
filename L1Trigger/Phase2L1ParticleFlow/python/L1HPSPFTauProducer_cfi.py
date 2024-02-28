import FWCore.ParameterSet.Config as cms

l1HPSPFTauEmuProducer = cms.EDProducer("L1HPSPFTauProducer",
                         srcL1PFCands = cms.InputTag("l1tLayer1:Puppi"),
                         nTaus        = cms.int32(16),
                         debug        = cms.bool(False),
                         useJets      = cms.bool(False),
			 srcL1PFJets = cms.InputTag("l1tPhase1JetCalibrator9x9trimmed:Phase1L1TJetFromPfCandidates")
                        )
