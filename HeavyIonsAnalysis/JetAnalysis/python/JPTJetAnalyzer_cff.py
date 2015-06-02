import FWCore.ParameterSet.Config as cms

JPTJetAnalyzer = cms.EDAnalyzer("HiJPTJetAnalyzer",

                                      jetTag = cms.InputTag("icPu5patJets"),
                                      matchTag = cms.untracked.InputTag("akPu3PFpatJets"),
                                      genjetTag = cms.InputTag("iterativeCone5HiGenJets"),
                                      eventInfoTag = cms.InputTag("generator"),
                                      isMC = cms.untracked.bool(False), 
                                      fillGenJets = cms.untracked.bool(False),
                                      rParam = cms.double(0.5),
                                      trackTag = cms.InputTag("hiTracks"),
                                      useQuality = cms.untracked.bool(True),
                                      trackQuality  = cms.untracked.string("highPurity"),
                                      useCentrality = cms.untracked.bool(False),
                                      doLifeTimeTagging = cms.untracked.bool(False),
                                      L1gtReadout = cms.InputTag("gtDigis"),

                                      doHiJetID = cms.untracked.bool(True),
                                      doStandardJetID = cms.untracked.bool(False),
                                      
                                      hltTrgResults = cms.untracked.string("TriggerResults::HLT"),
                                      hltTrgNames  = cms.untracked.vstring('HLT_HIMinBiasHfOrBSC_Core',
                                                                           'HLT_HIJet35U',
                                                                           'HLT_HIJet35U_Core',
                                                                           'HLT_HIJet50U_Core')
                                      )


#ic3JetAnalyzer = JPTJetAnalyzer.clone()
#ic3JetAnalyzer.jetTag = 'ic3patJets'
#ic3JetAnalyzer.genjetTag = 'iterativeCone3HiGenJets'

#ic4JetAnalyzer = JPTJetAnalyzer.clone()
#ic4JetAnalyzer.jetTag = 'ic4patJets'
#ic4JetAnalyzer.genjetTag = 'iterativeCone4HiGenJets'

#ic5JetAnalyzer = JPTJetAnalyzer.clone()
#ic5JetAnalyzer.jetTag = 'ic5patJets'
#ic4JetAnalyzer.genjetTag = 'iterativeCone5HiGenJets'

#ak3JetAnalyzer = JPTJetAnalyzer.clone()
#ak3JetAnalyzer.jetTag = 'ak3patJets'
#ak3JetAnalyzer.genjetTag = 'ak3HiGenJets'

#ak4JetAnalyzer = JPTJetAnalyzer.clone()
#ak4JetAnalyzer.jetTag = 'ak4patJets'
#ak4JetAnalyzer.genjetTag = 'ak4HiGenJets'

#ak5JetAnalyzer = JPTJetAnalyzer.clone()
#ak5JetAnalyzer.jetTag = 'ak5patJets'
#ak5JetAnalyzer.genjetTag = 'ak5HiGenJets'

#ak7JetAnalyzer = JPTJetAnalyzer.clone()
#ak7JetAnalyzer.jetTag = 'ak7patJets'
#ak7JetAnalyzer.genjetTag = 'ak7HiGenJets'

#kt4JetAnalyzer = JPTJetAnalyzer.clone()
#kt4JetAnalyzer.jetTag = 'kt4patJets'
#kt4JetAnalyzer.genjetTag = 'kt4HiGenJets'

#ic3PFJetAnalyzer = JPTJetAnalyzer.clone()
#ic3PFJetAnalyzer.jetTag = 'ic3PFpatJets'
#ic3PFJetAnalyzer.genjetTag = 'iterativeCone3HiGenJets'

#ic4PFJetAnalyzer = JPTJetAnalyzer.clone()
#ic4PFJetAnalyzer.jetTag = 'ic4PFpatJets'
#ic4PFJetAnalyzer.genjetTag = 'iterativeCone4HiGenJets'

#ic5PFJetAnalyzer = JPTJetAnalyzer.clone()
#ic5PFJetAnalyzer.jetTag = 'ic5PFpatJets'
#ic5PFJetAnalyzer.genjetTag = 'iterativeCone5HiGenJets'

#ak3PFJetAnalyzer = JPTJetAnalyzer.clone()
#ak3PFJetAnalyzer.jetTag = 'ak3PFpatJets'
#ak3PFJetAnalyzer.genjetTag = 'ak3HiGenJets'

#ak4PFJetAnalyzer = JPTJetAnalyzer.clone()
#ak4PFJetAnalyzer.jetTag = 'ak4PFpatJets'
#ak4PFJetAnalyzer.genjetTag = 'ak4HiGenJets'

#ak5PFJetAnalyzer = JPTJetAnalyzer.clone()
#ak5PFJetAnalyzer.jetTag = 'ak5PFpatJets'
#ak5PFJetAnalyzer.genjetTag = 'ak5HiGenJets'

#ak7PFJetAnalyzer = JPTJetAnalyzer.clone()
#ak7PFJetAnalyzer.jetTag = 'ak7PFpatJets'
#ak7PFJetAnalyzer.genjetTag = 'ak7HiGenJets'

#kt4PFJetAnalyzer = JPTJetAnalyzer.clone()
#kt4PFJetAnalyzer.jetTag = 'kt4PFpatJets'
#kt4PFJetAnalyzer.genjetTag = 'kt4HiGenJets'

#icPu5JPTJetAnalyzer = JPTJetAnalyzer.clone()
#icPu5JPTJetAnalyzer.jetTag = 'jpticPu5patJets'
#icPu5JPTJetAnalyzer.genjetTag = 'iterativeCone5HiGenJets'

