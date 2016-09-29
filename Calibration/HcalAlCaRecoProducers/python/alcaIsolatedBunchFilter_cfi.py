import FWCore.ParameterSet.Config as cms

# producer for alcaisolatedbunch (HCAL isolated bunch with Jet trigger)
AlcaIsolatedBunchFilter = cms.EDFilter("AlCaIsolatedBunchFilter",
                                       TriggerResultLabel= cms.InputTag("TriggerResults","","HLT"),
                                       ProcessName       = cms.string("HLT"),

                                       TriggerIsoBunch   = cms.vstring("HLT_ZeroBias_IsolatedBunches"),
                                       TriggerJet  = cms.vstring("HLT_AK8PFJet", 
                                                                 "HLT_AK8PFHT",
                                                                 "HLT_CaloJet",
                                                                 "HLT_HT", 
                                                                 "HLT_JetE", 
                                                                 "HLT_PFHT", 
                                                                 "HLT_DiPFJet",
                                                                 "HLT_PFJet", 
                                                                 "HLT_DiCentralPFJet", 
                                                                 "HLT_QuadPFJet", 
                                                                 "HLT_L1_TripleJet_VBF", 
                                                                 "HLT_QuadJet",
                                                                 "HLT_DoubleJet",
                                                                 "HLT_AK8DiPFJet",
                                                                 "HLT_AK4CaloJet", 
                                                                 "HLT_AK4PFJet"), 
                                       )
