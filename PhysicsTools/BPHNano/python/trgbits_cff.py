import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *



trgTable = cms.EDProducer( "TrgBitTableProducer",
                          hltresults = cms.InputTag("TriggerResults::HLT"),
                          l1results  = cms.InputTag("gtStage2Digis::RECO"),
                          #add interesting paths
                          paths      = cms.vstring(
                                             "HLT_Mu7_IP4",
                                             "HLT_Mu8_IP6",
                                             "HLT_Mu8_IP5",
                                             "HLT_Mu8_IP3",
                                             "HLT_Mu8p5_IP3p5",
                                             "HLT_Mu9_IP6",
                                             "HLT_Mu9_IP5",
                                             "HLT_Mu9_IP4",    
                                             "HLT_Mu10p5_IP3p5",
                                             "HLT_Mu12_IP6"
                                              ),
                           #add interesting seeds
                           seeds     = cms.vstring(
                                             "L1_SingleMu7er1p5",
                                             "L1_SingleMu8er1p5",
                                             "L1_SingleMu9er1p5",
                                             "L1_SingleMu10er1p5",
                                             "L1_SingleMu12er1p5",
                                             "L1_SingleMu22"
                                              ),
                            
)

trgTables = cms.Sequence(trgTable)



