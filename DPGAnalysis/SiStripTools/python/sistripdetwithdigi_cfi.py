import FWCore.ParameterSet.Config as cms

sistripdetwithdigi = cms.EDFilter('SiStripDetWithDigi',
                                  collectionName = cms.InputTag("siStripDigis","ZeroSuppressed"),  
                                  selectedModules = cms.untracked.vuint32()
)	
# foo bar baz
# iBaoMvxwYfpD1
# 4L5kOk1KNc1SM
