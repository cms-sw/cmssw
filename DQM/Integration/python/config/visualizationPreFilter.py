import FWCore.ParameterSet.Config as cms
#process.hltTriggerTypeFilter

TAG_HLTSEL = 'HLT_*'


hltHighLevel = cms.EDFilter("HLTHighLevel",
                            TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
                            HLTPaths = cms.vstring(           # provide list of HLT paths (or patterns) you want
                                TAG_HLTSEL
                                ),
                            eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
                            andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
                            throw = cms.bool(False)    # throw exception on unknown path names
                            )


hltfilter = cms.Sequence(hltHighLevel)

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters
filtersiPixelClusters = siPixelClusters.clone(
  src = "filtersiPixelDigis"
)

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis
filtersiPixelDigis = siPixelDigis.clone(
  InputLabel = "rawDataCollector"
)

import  HLTrigger.special.hltPixelActivityFilter_cfi
multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
    inputTag  = 'filtersiPixelClusters',
    minClusters = 10000,
    maxClusters = 50000
)

pixelClusterFilter = cms.Sequence(filtersiPixelDigis * filtersiPixelClusters * multFilter)

# process.hltfilter=cms.Path(process.hltHighLevel)


# process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# # 0=random, 1=physics, 2=calibration, 3=technical
# process.hltTriggerTypeFilter.SelectedTriggerType = TAG_TRIGGERTYPE

# process.triggertype=cms.Path(process.hltTriggerTypeFilter)

# # this is for filtering on L1 technical trigger bit
# process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
# process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
# process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
# process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string(TAG_L1TTSEL)
# process.techtrigger=cms.Path(process.hltLevel1GTSeed)

# #this is for filtering/tagging PhysDecl bit
# process.physdecl = cms.EDFilter("PhysDecl",
#                                      applyfilter = cms.untracked.bool(False),
#                                      debugOn = cms.untracked.bool(True),
#                                 # the following needs V00-01-19 of DPGAnalysis/Skims!!!
#                                      HLTriggerResults = cms.InputTag("TriggerResults","","HLT")

#                                     )
# process.Monitoring=cms.Path(process.physdecl)


# this is how the path was configured in the initial script from TB
#process.hltinspect
#process.hltTriggerTypeFilter
#RAWTODIGI
#process.hltLevel1GTSeed
#RECO
#process.l1GtTriggerMenuLite+process.beamsplash+process.physdecl+process.dcsstatus
