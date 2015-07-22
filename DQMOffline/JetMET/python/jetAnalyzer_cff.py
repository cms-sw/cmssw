import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetAnalyzer_cfi import *

#jet correctors defined in JetMETDQMOfflineSource python file

jetDQMAnalyzerSequence = cms.Sequence(jetDQMAnalyzerAk4CaloCleaned
                                      *jetDQMAnalyzerAk4PFUncleaned*jetDQMAnalyzerAk4PFCleaned
                                      *jetDQMAnalyzerAk4PFCHSCleaned
                                   )

jetDQMAnalyzerSequenceCosmics = cms.Sequence(jetDQMAnalyzerAk4CaloUncleaned)

jetDQMAnalyzerSequenceMiniAOD = cms.Sequence(jetDQMAnalyzerAk4PFCHSUncleanedMiniAOD*jetDQMAnalyzerAk4PFCHSCleanedMiniAOD)

jetDQMAnalyzerSequenceHI = cms.Sequence(jetDQMAnalyzerIC5CaloHIUncleaned
                                        * jetDQMMatchAkPu3CaloAkVs3Calo
                                        * jetDQMMatchAkPu3PFAkVs3PF
                                        * jetDQMMatchAkPu3CaloAkPu3PF
                                        * jetDQMMatchAkVs3CaloAkVs3PF

                                        * jetDQMMatchAkPu4CaloAkVs4Calo
                                        * jetDQMMatchAkPu4PFAkVs4PF
                                        * jetDQMMatchAkPu4CaloAkPu4PF
                                        * jetDQMMatchAkVs4CaloAkVs4PF

                                        * jetDQMMatchAkPu5CaloAkVs5Calo
                                        * jetDQMMatchAkPu5PFAkVs5PF
                                        * jetDQMMatchAkPu5CaloAkPu5PF
                                        * jetDQMMatchAkVs5CaloAkVs5PF
                                        
                                        * jetDQMAnalyzerAkPU3Calo
                                        * jetDQMAnalyzerAkPU4Calo
                                        * jetDQMAnalyzerAkPU5Calo
                                        
                                        * jetDQMAnalyzerAkPU3PF
                                        * jetDQMAnalyzerAkPU4PF
                                        * jetDQMAnalyzerAkPU5PF

                                        #* jetDQMAnalyzerAkVs2Calo	   
                                        * jetDQMAnalyzerAkVs3Calo	   
                                        * jetDQMAnalyzerAkVs4Calo	   
                                        * jetDQMAnalyzerAkVs5Calo	   
                                        #* jetDQMAnalyzerAkVs6Calo
                                        #* jetDQMAnalyzerAkVs7Calo
                                        
                                        #* jetDQMAnalyzerAkVs2PF
                                        * jetDQMAnalyzerAkVs3PF
                                        * jetDQMAnalyzerAkVs4PF	   
                                        * jetDQMAnalyzerAkVs5PF
                                        #* jetDQMAnalyzerAkVs6PF	   
                                        #* jetDQMAnalyzerAkVs7PF

                                        #* jetDQMAnalyzerAk3CaloCleaned
                                        #* jetDQMAnalyzerAk4CaloCleaned
                                        #* jetDQMAnalyzerAk5CaloCleaned
                                        #* jetDQMAnalyzerAk3PFCleaned
                                        #* jetDQMAnalyzerAk4PFCleaned
                                        #* jetDQMAnalyzerAk5PFCleaned                                        
)

