import FWCore.ParameterSet.Config as cms

SiStripSpyEventMatcher = cms.EDFilter(
    "SiStripSpyEventMatcherModule",
    FilterNonMatchingEvents = cms.bool(True),
    MergeData = cms.bool(True),
    PrimaryEventRawDataTag = cms.InputTag('source'),
    SpyTotalEventCountersTag = cms.InputTag('SiStripSpyUnpacker','TotalEventCount'),
    SpyL1ACountersTag = cms.InputTag('SiStripSpyUnpacker','L1ACount'),
    SpyAPVAddressesTag = cms.InputTag('SiStripSpyDigiConverter','APVAddress'),
    RawSpyDataTag = cms.InputTag('source'),
    SpyScopeDigisTag = cms.InputTag('SiStripSpyUnpacker','ScopeRawDigis'),
    SpyPayloadDigisTag = cms.InputTag('SiStripSpyDigiConverter','Payload'),
    SpyReorderedDigisTag = cms.InputTag('SiStripSpyDigiConverter','Reordered'),
    SpyVirginRawDigisTag = cms.InputTag('SiStripSpyDigiConverter','VirginRaw'),
    SpySource = cms.SecSource(
      "EmbeddedRootSource",
      fileNames = cms.untracked.vstring(
        'SpyFileNameWhichNeedsToBeSet SiStripSpyEventMatcher.SpySource.fileNames'
        ),
        sequential = cms.untracked.bool(True),
      ),
    CounterDiffMaxAllowed = cms.uint32(100)
    )

