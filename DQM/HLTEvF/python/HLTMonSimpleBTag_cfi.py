import FWCore.ParameterSet.Config as cms

hltMonSimpleBTag = cms.EDAnalyzer("HLTMonSimpleBTag",
                                  ptMax = cms.untracked.double(300.0),
                                  ptMin = cms.untracked.double(0.0),
                                  Nbins = cms.untracked.uint32(30),
                                  dRMatch = cms.untracked.double(0.3),

                                  #string combinations, passes two triggers. Both are saved separately, but in addition the triggers are matched and an efficiency is calculated (objects are dR matched in this case, matching angle is defined above)
#                                  BTagMu_DiJet20_Mu5
#                                  BTagMu_DiJet60_Mu7
#                                  BTagMu_DiJet80_Mu9
#                                  BTagMu_DiJet100_Mu9
# picked a random L1 term to compare to: 
                                 
                                  filters = cms.VPSet(
                                                      cms.PSet(name=cms.string("hltBSoftMuonDiJet20Mu5SelL3FilterByDR"),
                                                               refname=cms.string("hltBDiJet20Central")
                                                               ),
                                                      cms.PSet(name=cms.string("hltBSoftMuonDiJet60Mu7SelL3FilterByDR"),
                                                               refname=cms.string("hltBDiJet20Central")
                                                               ),
                                                      cms.PSet(name=cms.string("hltBSoftMuonDiJet80Mu9SelL3FilterByDR"),
                                                               refname=cms.string("hltBDiJet20Central")
                                                               ),
                                                      cms.PSet(name=cms.string("hltBSoftMuonDiJet80Mu9SelL3FilterByDR"),
                                                               refname=cms.string("hltBSoftMuonDiJet60Mu7SelL3FilterByDR")
                                                               ),
                                                      cms.PSet(name=cms.string("hltBSoftMuonDiJet60Mu7SelL3FilterByDR"),
                                                               refname=cms.string("hltBSoftMuonDiJet20Mu5SelL3FilterByDR")
                                                               )
                                      ),
                                  # data best guess
                                  triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT")
                                                      
)
                                  
