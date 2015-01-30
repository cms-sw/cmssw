import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetAnalyzer_cfi import *

#jet correctors defined in JetMETDQMOfflineSource python file

jetDQMAnalyzerSequence = cms.Sequence(jetDQMAnalyzerAk4CaloCleaned
                                      *jetDQMAnalyzerAk4PFUncleaned*jetDQMAnalyzerAk4PFCleaned
                                      *jetDQMAnalyzerAk4PFCHSCleaned
                                   )

jetDQMAnalyzerSequenceCosmics = cms.Sequence(jetDQMAnalyzerAk4CaloUncleaned)

jetDQMAnalyzerSequenceHI = cms.Sequence(jetDQMAnalyzerIC5CaloHIUncleaned
                                        * jetDQMAnalyzerAkPU3Calo
                                        * jetDQMAnalyzerAkPU3PF
                                        * jetDQMAnalyzerAkVs3Calo	   
                                        * jetDQMAnalyzerAkVs3PF                                        
                                        )

jetDQMAnalyzerSequenceMiniAOD = cms.Sequence(jetDQMAnalyzerAk4PFCHSUncleanedMiniAOD*jetDQMAnalyzerAk4PFCHSCleanedMiniAOD)

