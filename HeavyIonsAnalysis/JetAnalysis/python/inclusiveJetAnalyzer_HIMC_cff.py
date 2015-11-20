import FWCore.ParameterSet.Config as cms

inclusiveJetAnalyzer = cms.EDAnalyzer("HiInclusiveJetAnalyzer",
                                      jetTag = cms.InputTag("icPu5patJets"),
                                      genjetTag = cms.InputTag("iterativeCone5HiGenJets"),
                                      isMC = cms.untracked.bool(True), 
				      useHepMC = cms.untracked.bool(True),
                                      useCentrality = cms.untracked.bool(True),
                                      L1gtReadout = cms.InputTag("gtDigis"),
                                      hltTrgResults = cms.untracked.string("TriggerResults::HLT"),
                                      hltTrgNames  = cms.untracked.vstring('HLT_HIMinBiasHfOrBSC_Core',
                                                                           'HLT_HIJet35U',
                                                                           'HLT_HIJet35U_Core',
                                                                           'HLT_HIJet50U_Core')
                                      )


ic3JetAnalyzer = inclusiveJetAnalyzer.clone()
ic3JetAnalyzer.jetTag = 'ic3patJets'
ic3JetAnalyzer.genjetTag = 'iterativeCone3HiGenJets'

ic4JetAnalyzer = inclusiveJetAnalyzer.clone()
ic4JetAnalyzer.jetTag = 'ic4patJets'
ic4JetAnalyzer.genjetTag = 'iterativeCone4HiGenJets'

ic5JetAnalyzer = inclusiveJetAnalyzer.clone()
ic5JetAnalyzer.jetTag = 'ic5patJets'
ic4JetAnalyzer.genjetTag = 'iterativeCone5HiGenJets'

ak3JetAnalyzer = inclusiveJetAnalyzer.clone()
ak3JetAnalyzer.jetTag = 'ak3patJets'
ak3JetAnalyzer.genjetTag = 'ak3HiGenJets'

ak4JetAnalyzer = inclusiveJetAnalyzer.clone()
ak4JetAnalyzer.jetTag = 'ak4patJets'
ak4JetAnalyzer.genjetTag = 'ak4HiGenJets'

ak5JetAnalyzer = inclusiveJetAnalyzer.clone()
ak5JetAnalyzer.jetTag = 'ak5patJets'
ak5JetAnalyzer.genjetTag = 'ak5HiGenJets'

ak7JetAnalyzer = inclusiveJetAnalyzer.clone()
ak7JetAnalyzer.jetTag = 'ak7patJets'
ak7JetAnalyzer.genjetTag = 'ak7HiGenJets'

kt4JetAnalyzer = inclusiveJetAnalyzer.clone()
kt4JetAnalyzer.jetTag = 'kt4patJets'
kt4JetAnalyzer.genjetTag = 'kt4HiGenJets'

ic3PFJetAnalyzer = inclusiveJetAnalyzer.clone()
ic3PFJetAnalyzer.jetTag = 'ic3PFpatJets'
ic3PFJetAnalyzer.genjetTag = 'iterativeCone3HiGenJets'

ic4PFJetAnalyzer = inclusiveJetAnalyzer.clone()
ic4PFJetAnalyzer.jetTag = 'ic4PFpatJets'
ic4PFJetAnalyzer.genjetTag = 'iterativeCone4HiGenJets'

ic5PFJetAnalyzer = inclusiveJetAnalyzer.clone()
ic5PFJetAnalyzer.jetTag = 'ic5PFpatJets'
ic5PFJetAnalyzer.genjetTag = 'iterativeCone5HiGenJets'

ak3PFJetAnalyzer = inclusiveJetAnalyzer.clone()
ak3PFJetAnalyzer.jetTag = 'ak3PFpatJets'
ak3PFJetAnalyzer.genjetTag = 'ak3HiGenJets'

ak4PFJetAnalyzer = inclusiveJetAnalyzer.clone()
ak4PFJetAnalyzer.jetTag = 'ak4PFpatJets'
ak4PFJetAnalyzer.genjetTag = 'ak4HiGenJets'

ak5PFJetAnalyzer = inclusiveJetAnalyzer.clone()
ak5PFJetAnalyzer.jetTag = 'ak5PFpatJets'
ak5PFJetAnalyzer.genjetTag = 'ak5HiGenJets'

ak7PFJetAnalyzer = inclusiveJetAnalyzer.clone()
ak7PFJetAnalyzer.jetTag = 'ak7PFpatJets'
ak7PFJetAnalyzer.genjetTag = 'ak7HiGenJets'

icPu5JPTJetAnalyzer = inclusiveJetAnalyzer.clone()
icPu5JPTJetAnalyzer.jetTag = 'jpticPu5patJets'
icPu5JPTJetAnalyzer.genjetTag = 'iterativeCone5HiGenJets'

allJetAnalyzers = cms.Sequence(                         inclusiveJetAnalyzer
                                                        *ic3JetAnalyzer
                                                        *ic4JetAnalyzer
                                                        *ic5JetAnalyzer
                                                        *ak3JetAnalyzer
                                                        *ak4JetAnalyzer
                                                        *ak5JetAnalyzer
                                                        *kt4JetAnalyzer
                                                        *ak7JetAnalyzer
                                                        *ic3PFJetAnalyzer
                                                        *ic4PFJetAnalyzer
                                                        *ic5PFJetAnalyzer
                                                        *ak3PFJetAnalyzer
                                                        *ak4PFJetAnalyzer
                                                        *ak5PFJetAnalyzer
                                                        *ak7PFJetAnalyzer
                                                        *icPu5JPTJetAnalyzer
                                                        )



