# Configuration for merging spy rawAndCounters data with matching primaryRaw
#============================================================================
import FWCore.ParameterSet.Config as cms

process = cms.Process('SPYEVENTMATCHING')

#source of normal event data
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
       #penultimate file in primary dataset 121835 (RandomTriggers with known matching spy event)
       '/store/data/BeamCommissioning09/RandomTriggers/RAW/v1/000/121/835/029D594D-77D5-DE11-9F3B-00304867342C.root'
       )
    )

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

#process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))
process.load('DQM.SiStripCommon.MessageLogger_cfi')

# --- Conditions data ---
# Find the appropriate Global Tags at
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR09_P_V8_34X::All'

#merger module
process.load('DQM.SiStripMonitorHardware.SiStripSpyEventMatcher_cfi')
process.SiStripSpyEventMatcher.SpySource.fileNames = cms.untracked.vstring(
    'rfio:/castor/cern.ch/user/a/amagnan/SpyEvts/121834/spyunpackRawAndCountersOutput_1.root',
    'rfio:/castor/cern.ch/user/a/amagnan/SpyEvts/121834/spyunpackRawAndCountersOutput_2.root',
    'rfio:/castor/cern.ch/user/a/amagnan/SpyEvts/121834/spyunpackRawAndCountersOutput_3.root',
    'rfio:/castor/cern.ch/user/a/amagnan/SpyEvts/121834/spyunpackRawAndCountersOutput_4.root',
    'rfio:/castor/cern.ch/user/a/amagnan/SpyEvts/121834/spyunpackRawAndCountersOutput_5.root',
    'rfio:/castor/cern.ch/user/a/amagnan/SpyEvts/121834/spyunpackRawAndCountersOutput_6.root',
    'rfio:/castor/cern.ch/user/a/amagnan/SpyEvts/121834/spyunpackRawAndCountersOutput_7.root',
    'rfio:/castor/cern.ch/user/a/amagnan/SpyEvts/121834/spyunpackRawAndCountersOutput_8.root',
    'rfio:/castor/cern.ch/user/a/amagnan/SpyEvts/121834/spyunpackRawAndCountersOutput_9.root'
    )
process.SiStripSpyEventMatcher.FilterNonMatchingEvents = cms.bool(True)
process.SiStripSpyEventMatcher.MergeData = cms.bool(True)
process.SiStripSpyEventMatcher.PrimaryEventRawDataTag = cms.InputTag('source')
process.SiStripSpyEventMatcher.SpyTotalEventCountersTag = cms.InputTag('SiStripSpyUnpacker','TotalEventCount')
process.SiStripSpyEventMatcher.SpyL1ACountersTag = cms.InputTag('SiStripSpyUnpacker','L1ACount')
process.SiStripSpyEventMatcher.SpyAPVAddressesTag = cms.InputTag('SiStripSpyDigiConverter','APVAddress')
process.SiStripSpyEventMatcher.RawSpyDataTag = cms.InputTag('source')
process.SiStripSpyEventMatcher.SpyScopeDigisTag = cms.InputTag('SiStripSpyUnpacker','ScopeRawDigis')
process.SiStripSpyEventMatcher.SpyPayloadDigisTag = cms.InputTag('SiStripSpyDigiConverter','Payload')
process.SiStripSpyEventMatcher.SpyReorderedDigisTag = cms.InputTag('SiStripSpyDigiConverter','Reordered')
process.SiStripSpyEventMatcher.SpyVirginRawDigisTag = cms.InputTag('SiStripSpyDigiConverter','VirginRaw')

# ---- Path
process.p = cms.Path(
    process.SiStripSpyEventMatcher
    )

process.output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("SpyMatchedEvents.root"),
    outputCommands = cms.untracked.vstring(
        'keep *'
        #'drop *',
        #'keep *_source_*_*'
        ),
    # Select only events that pass the spy matching filter
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p') 
        )
    )

process.e = cms.EndPath( process.output )
