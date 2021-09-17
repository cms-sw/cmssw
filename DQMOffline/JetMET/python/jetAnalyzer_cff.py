import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetAnalyzer_cfi import *

#jet correctors defined in JetMETDQMOfflineSource python file

jetDQMAnalyzerSequence = cms.Sequence(jetDQMAnalyzerAk4CaloCleaned
                                      *jetDQMAnalyzerAk4PFUncleaned*jetDQMAnalyzerAk4PFCleaned
                                      *jetDQMAnalyzerAk4PFCHSCleaned
                                   )

jetDQMAnalyzerSequenceCosmics = cms.Sequence(jetDQMAnalyzerAk4CaloUncleaned)

jetDQMAnalyzerSequenceMiniAOD = cms.Sequence(jetDQMAnalyzerAk4PFCHSUncleanedMiniAOD*jetDQMAnalyzerAk4PFCHSCleanedMiniAOD*jetDQMAnalyzerAk8PFPUPPICleanedMiniAOD*jetDQMAnalyzerAk4PFCHSPuppiCleanedMiniAOD)

jetDQMAnalyzerSequenceHI = cms.Sequence(jetDQMAnalyzerIC5CaloHIUncleaned
                                        * jetDQMMatchAkPu3CaloAkPu3PF
                                        * jetDQMMatchAkPu4CaloAkPu4PF
                                        * jetDQMMatchAkPu5CaloAkPu5PF

                                        * jetDQMAnalyzerAkPU3Calo
                                        * jetDQMAnalyzerAkPU4Calo
                                        * jetDQMAnalyzerAkPU5Calo
                                        
                                        * jetDQMAnalyzerAkPU3PF
                                        * jetDQMAnalyzerAkPU4PF
                                        * jetDQMAnalyzerAkPU5PF
                                     
                                        * jetDQMAnalyzerAkCs3PF
                                        * jetDQMAnalyzerAkCs4PF
)

_jetDQMAnalyzerSequenceHI = cms.Sequence(jetDQMMatchAkPu4CaloAkPu4PF
        * jetDQMAnalyzerAkPU4Calo
        * jetDQMAnalyzerAkPU3PF
        * jetDQMAnalyzerAkPU4PF
        * jetDQMAnalyzerAkCs4PF
)
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith( jetDQMAnalyzerSequence, _jetDQMAnalyzerSequenceHI )
pp_on_AA.toModify( jetDQMAnalyzerAkPU4Calo, srcVtx = cms.untracked.InputTag("offlinePrimaryVertices") )
pp_on_AA.toModify( jetDQMAnalyzerAkPU3PF, srcVtx = cms.untracked.InputTag("offlinePrimaryVertices") )
pp_on_AA.toModify( jetDQMAnalyzerAkPU4PF, srcVtx = cms.untracked.InputTag("offlinePrimaryVertices") )
pp_on_AA.toModify( jetDQMAnalyzerAkCs4PF, srcVtx = cms.untracked.InputTag("offlinePrimaryVertices") )

