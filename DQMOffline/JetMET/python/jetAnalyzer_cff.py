import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetAnalyzer_cfi import *

jetDQMAnalyzerSequence = cms.Sequence(jetDQMAnalyzerAk4CaloUncleaned*jetDQMAnalyzerAk4CaloCleaned
#                                   *jetDQMAnalyzerAk4JPTCleaned
                                   *jetDQMAnalyzerAk4PFUncleaned*jetDQMAnalyzerAk4PFCleaned
                                   )

jetDQMAnalyzerSequenceCosmics = cms.Sequence(jetDQMAnalyzerAk4CaloUncleaned)

jetDQMAnalyzerSequenceHI = cms.Sequence(jetDQMAnalyzerIC5CaloHIUncleaned
                                        * jetDQMAnalyzerAkPU3Calo
                                        #* jetDQMAnalyzerAkPU4Calo
                                        #* jetDQMAnalyzerAkPU5Calo
                                        
                                        * jetDQMAnalyzerAkPU3PF
                                        #* jetDQMAnalyzerAkPU4PF
                                        #* jetDQMAnalyzerAkPU5PF

                                        #* jetDQMAnalyzerAkVs2Calo	   
                                        * jetDQMAnalyzerAkVs3Calo	   
                                        #* jetDQMAnalyzerAkVs4Calo	   
                                        #* jetDQMAnalyzerAkVs5Calo	   
                                        #* jetDQMAnalyzerAkVs6Calo
                                        #* jetDQMAnalyzerAkVs7Calo
                                        
                                        #* jetDQMAnalyzerAkVs2PF
                                        * jetDQMAnalyzerAkVs3PF
                                        #* jetDQMAnalyzerAkVs4PF	   
                                        #* jetDQMAnalyzerAkVs5PF
                                        #* jetDQMAnalyzerAkVs6PF	   
                                        #* jetDQMAnalyzerAkVs7PF

                                        #* jetDQMAnalyzerAk3CaloCleaned
                                        #* jetDQMAnalyzerAk4CaloCleaned
                                        #* jetDQMAnalyzerAk5CaloCleaned
                                        #* jetDQMAnalyzerAk3PFCleaned
                                        #* jetDQMAnalyzerAk4PFCleaned
                                        #* jetDQMAnalyzerAk5PFCleaned                                        
)
