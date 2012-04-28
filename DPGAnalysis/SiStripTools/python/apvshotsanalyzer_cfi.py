import FWCore.ParameterSet.Config as cms

apvshotsanalyzer = cms.EDAnalyzer('APVShotsAnalyzer',
                                  digiCollection = cms.InputTag("siStripDigis","ZeroSuppressed"),
                                  historyProduct = cms.InputTag("consecutiveHEs"),
                                  apvPhaseCollection = cms.InputTag("APVPhases"),
                                  phasePartition = cms.untracked.string("All"),
                                  zeroSuppressed = cms.untracked.bool(True),
                                  mapSuffix = cms.string(""),
                                  useCabling = cms.untracked.bool(False)
)	
