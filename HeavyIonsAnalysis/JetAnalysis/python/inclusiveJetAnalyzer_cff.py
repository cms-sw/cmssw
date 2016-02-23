import FWCore.ParameterSet.Config as cms

inclusiveJetAnalyzer = cms.EDAnalyzer("HiInclusiveJetAnalyzer",

                                      jetTag = cms.InputTag("icPu5patJets"),
                                      jetPtMin = cms.double(5.0),
                                      matchTag = cms.untracked.InputTag("akPu3PFpatJets"),
                                      genjetTag = cms.InputTag("iterativeCone5HiGenJets"),
                                      eventInfoTag = cms.InputTag("generator"),
                                      isMC = cms.untracked.bool(False), 
                                      fillGenJets = cms.untracked.bool(False),
                                      rParam = cms.double(0.5),
                                      trackTag = cms.InputTag("hiTracks"),
                                      useHepMC = cms.untracked.bool(False),
                                      useQuality = cms.untracked.bool(True),
                                      trackQuality  = cms.untracked.string("highPurity"),
                                      useCentrality = cms.untracked.bool(False),
                                      doLifeTimeTagging = cms.untracked.bool(False),
                                      L1gtReadout = cms.InputTag("gtDigis"),

                                      doHiJetID = cms.untracked.bool(True),
                                      doStandardJetID = cms.untracked.bool(False),
                                      doSubEvent = cms.untracked.bool(False),
                                      
                                      hltTrgResults = cms.untracked.string("TriggerResults::HLT"),
                                      hltTrgNames  = cms.untracked.vstring('HLT_HIMinBiasHfOrBSC_Core',
                                                                           'HLT_HIJet35U',
                                                                           'HLT_HIJet35U_Core',
                                                                           'HLT_HIJet50U_Core')
                                      )
