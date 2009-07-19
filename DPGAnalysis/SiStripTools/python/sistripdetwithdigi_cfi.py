import FWCore.ParameterSet.Config as cms

sistripdetwithdigi = cms.EDAnalyzer('SiStripDetWithDigi',
                                collectionName = cms.InputTag("siStripDigis","ZeroSuppressed"),  
                                selectedModules = cms.untracked.vuint32()
)	
