import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetAnalyzer_cfi import *

#jet correctors defined in JetMETDQMOfflineSource python file

jetDQMAnalyzerSequence = cms.Sequence(jetDQMAnalyzerAk4CaloCleaned
                                      *jetDQMAnalyzerAk4PFUncleaned*jetDQMAnalyzerAk4PFCleaned
                                      *jetDQMAnalyzerAk4PFCHSCleaned
                                   )

jetDQMAnalyzerSequenceCosmics = cms.Sequence(jetDQMAnalyzerAk4CaloUncleaned)

jetDQMAnalyzerSequenceMiniAOD = cms.Sequence(jetDQMAnalyzerAk4PFCHSUncleanedMiniAOD*jetDQMAnalyzerAk4PFCHSCleanedMiniAOD*jetDQMAnalyzerAk8PFPUPPICleanedMiniAOD*jetDQMAnalyzerAk4PFCHSPuppiCleanedMiniAOD)

jetDQMAnalyzerSequenceHI = cms.Sequence(jetDQMMatchAkPu4CaloAkPu4PF

                                        * jetDQMAnalyzerAkPU4Calo
                                        
                                        * jetDQMAnalyzerAkPU3PF
                                        * jetDQMAnalyzerAkPU4PF
                                     
                                        * jetDQMAnalyzerAkCs4PF
)

