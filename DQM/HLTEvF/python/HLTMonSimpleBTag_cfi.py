import FWCore.ParameterSet.Config as cms

hltMonSimpleBTag = cms.EDAnalyzer("HLTMonSimpleBTag",
                                  ptMax = cms.untracked.double(100.0),
                                  ptMin = cms.untracked.double(0.0),
                                  Nbins = cms.untracked.uint32(100),
                                  dRMatch = cms.untracked.double(0.2),

                                  #string combinations, passes two triggers. Both are saved separately, but in addition the triggers are matched and an efficiency is calculated (objects are dR matched in this case, matching angle is defined above)
#                                  BTagMu_DiJet20_Mu5
#                                  BTagMu_DiJet60_Mu7
#                                  BTagMu_DiJet80_Mu9
#                                  BTagMu_DiJet100_Mu9

                                  filters = cms.VPSet(cms.PSet(name=cms.string("hltL1sBTagMuDiJet10U"),
                                                               refname=cms.string("hltL1sZeroBias")
                                                               ),
                                                      cms.PSet(name=cms.string("trig1"),
                                                               refname=cms.string("trig2")
                                                               ),
                                                      cms.PSet(name=cms.string("trig2"),
                                                               refname=cms.string("trig3")
                                                               ),
                                                      cms.PSet(name=cms.string("BTagMu_DiJet20_Mu5"),
                                                               refname=cms.string("DiJet20")
                                                               ),
                                                      cms.PSet(name=cms.string("BTagMu_DiJet60_Mu7"),
                                                               refname=cms.string("DiJet20")
                                                               ),
                                                      cms.PSet(name=cms.string("BTagMu_DiJet80_Mu9"),
                                                               refname=cms.string("DiJet20")
                                                               ),
                                                      cms.PSet(name=cms.string("BTagMu_DiJet100_Mu9"),
                                                               refname=cms.string("DiJet20")
                                                               )
                                                      ),
                                  # here one can define a combination between a trigger term and HLT monitoring objects. Again the matching is done using the deltaR requirement above. Can currently only deal with objects of type reco::Jet, reco::Muon or reco::TagInfos
                                  # WARNING: CURRENTLY NOT FULLY IMPLEMENTED!!!
                                  hltobjects = cms.VPSet(cms.PSet(name=cms.string("BTagMu_DiJet100_Mu9"),
                                                                  refname=cms.string("hltBSoftMuonDiJet60L25Jets"),
                                                                  ),
                                                         cms.PSet(name=cms.string("BTagMu_DiJet100_Mu9"),
                                                                  refname=cms.string("hltBSoftMuonDiJet60L25Jets")
                                                                  )
                                                         ),
                                  # data best guess
                                  triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT")
                                                      
)
                                  
