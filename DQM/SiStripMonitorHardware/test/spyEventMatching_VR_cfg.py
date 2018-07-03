# Configuration for merging spy rawAndCounters data with matching primaryRaw
#============================================================================
import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process('SPYEVENTMATCHING')

#source of normal event data
process.source = cms.Source(
    'PoolSource',
    fileNames = cms.untracked.vstring(
       #penultimate file in primary dataset 121835 (RandomTriggers with known matching spy event)
	'file:/eos/user/j/jblee/MainStream/298/269/00000/D04CCB24-4862-E711-92F2-02163E011F09.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/data/Commissioning2015/RAW/v1/000/234/846/00000/3E397F5D-25B9-E411-801B-02163E011CD9.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/data/Commissioning2015/RAW/v1/000/234/848/00000/54F96357-25B9-E411-B562-02163E0128CE.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/data/Commissioning2015/RAW/v1/000/234/856/00000/082589A2-25B9-E411-ABB7-02163E012BDD.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/data/Commissioning2015/RAW/v1/000/234/858/00000/064F08A5-10BB-E411-A2C0-02163E0123EF.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/data/Commissioning2015/RAW/v1/000/234/874/00000/06B0CB9B-77BB-E411-9959-02163E0125EB.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/data/Commissioning2015/RAW/v1/000/234/874/00000/22357546-2AB9-E411-964B-02163E01206E.root',
#'file:/eos/cms/store/group/dpg_tracker_strip/tracker/Online/store/data/Commissioning2015/RAW/v1/000/234/874/00000/66D33846-2AB9-E411-9CC8-02163E0122DB.root',
       )
    )

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

#process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))
process.load('DQM.SiStripCommon.MessageLogger_cfi')
process.load('EventFilter.SiStripRawToDigi.SiStripDigis_cfi')

# --- Conditions data ---
# Find the appropriate Global Tags at
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
process.load("Configuration.Geometry.GeometryRecoDB_cff")


#merger module
process.load('DQM.SiStripMonitorHardware.SiStripSpyEventMatcher_cfi')
process.SiStripSpyEventMatcher.SpySource.fileNames = cms.untracked.vstring(
#    'file:/eos/cms/store/user/jblee/SpyFEDemulated234824.root'.
     'file:/eos/cms/store/user/jblee//SpyRawToDigis298270_TEST.root'
    )
process.SiStripSpyEventMatcher.FilterNonMatchingEvents = cms.bool(True)
process.SiStripSpyEventMatcher.MergeData = cms.bool(True)
process.SiStripSpyEventMatcher.PrimaryEventRawDataTag = cms.InputTag('rawDataCollector')
process.SiStripSpyEventMatcher.SpyTotalEventCountersTag = cms.InputTag('SiStripSpyUnpacker','TotalEventCount')
process.SiStripSpyEventMatcher.SpyL1ACountersTag = cms.InputTag('SiStripSpyUnpacker','L1ACount')
process.SiStripSpyEventMatcher.SpyAPVAddressesTag = cms.InputTag('SiStripSpyDigiConverter','APVAddress')
process.SiStripSpyEventMatcher.RawSpyDataTag = cms.InputTag('rawDataCollector')
process.SiStripSpyEventMatcher.SpyScopeDigisTag = cms.InputTag('SiStripSpyUnpacker','ScopeRawDigis')
process.SiStripSpyEventMatcher.SpyPayloadDigisTag = cms.InputTag('SiStripSpyDigiConverter','SpyPayload')
process.SiStripSpyEventMatcher.SpyReorderedDigisTag = cms.InputTag('SiStripSpyDigiConverter','SpyReordered')
process.SiStripSpyEventMatcher.SpyVirginRawDigisTag = cms.InputTag('SiStripSpyDigiConverter','SpyVirginRaw')

# ---- Path
process.p = cms.Path(
    process.siStripDigis*
    process.SiStripSpyEventMatcher
    )

process.output = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("SpyMatchedEvents298270_TEST.root"),
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
