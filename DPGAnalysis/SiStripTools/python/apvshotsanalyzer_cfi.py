import FWCore.ParameterSet.Config as cms

apvshotsanalyzer = cms.EDAnalyzer('APVShotsAnalyzer',
                                  digiCollection = cms.InputTag("siStripDigis","ZeroSuppressed")                    
)	
