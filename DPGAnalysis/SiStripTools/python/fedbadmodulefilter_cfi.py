import FWCore.ParameterSet.Config as cms

fedbadmodulefilter = cms.EDFilter('FEDBadModuleFilter',
                                  collectionName = cms.InputTag("siStripDigis"),	
                                  badModThr = cms.uint32(35000)
                                  )
