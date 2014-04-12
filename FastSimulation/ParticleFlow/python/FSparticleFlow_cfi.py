import FWCore.ParameterSet.Config as cms

FSparticleFlow = cms.EDProducer("FSPFProducer",
                                # PFCandidate label
                                pfCandidates = cms.InputTag("particleFlowTmp"),
                                # parameters of the linear dependence between neutral and charged hadrons in PF:
                                par1 = cms.double(0.145),
                                par2 = cms.double(0.0031),
                                # eta-dependent thresholds:
                                barrel_th = cms.double(0.8), 
                                middle_th = cms.double(1.1), 
                                endcap_th = cms.double(2.4), 
                                # specific for HF candidates:
                                pfPatchInHF = cms.bool(False),
                                EM_HF_ScaleFactor = cms.vdouble(0.675, 0.55), # eta-dependent scale factor for EM tower energy
                                HF_Ratio = cms.double(1.7) # ratio of the number of EM/had towers in FullSim
                                )



