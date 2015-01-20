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
                                        * jetDQMAnalyzerAkPU4Calo
                                        * jetDQMAnalyzerAkPU5Calo
                                        
                                        * jetDQMAnalyzerAkPU3PF
                                        * jetDQMAnalyzerAkPU4PF
                                        * jetDQMAnalyzerAkPU5PF

                                        * jetDQMAnalyzerAkVs2Calo	   
                                        * jetDQMAnalyzerAkVs3Calo	   
                                        * jetDQMAnalyzerAkVs4Calo	   
                                        * jetDQMAnalyzerAkVs5Calo	   
                                        #* jetDQMAnalyzerAkVs6Calo
                                        #* jetDQMAnalyzerAkVs7Calo
                                        
                                        #* jetDQMAnalyzerAkVs2PF
                                        * jetDQMAnalyzerAkVs3PF                                        
                                        * jetDQMAnalyzerAkVs4PF
                                        * jetDQMAnalyzerAkVs5PF
                                        )

jetDQMAnalyzerSequenceMiniAOD = cms.Sequence(jetDQMAnalyzerAk4PFCHSUncleanedMiniAOD*jetDQMAnalyzerAk4PFCHSCleanedMiniAOD)

