import FWCore.ParameterSet.Config as cms

sistripdetwithdigi = cms.EDFilter('SiStripDetWithDigi',
                                  collectionName = cms.InputTag("siStripDigis","ZeroSuppressed"),  
                                  selectedModules = cms.untracked.vuint32()
)	
