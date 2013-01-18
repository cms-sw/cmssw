import FWCore.ParameterSet.Config as cms

apvshotsanalyzer = cms.EDAnalyzer('APVShotsAnalyzer',
                                  digiCollection = cms.InputTag("siStripDigis","ZeroSuppressed"),
                                  zeroSuppressed = cms.untracked.bool(True),
                                  mapSuffix = cms.string(""),
                                  useCabling = cms.untracked.bool(False)
)	
