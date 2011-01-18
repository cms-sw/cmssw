import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQM.BeamMonitor.BeamMonitor_cff")
process.load("DQM.BeamMonitor.BeamMonitorBx_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/BAEF02C0-0BED-DE11-9EBA-00261894392F.root',
'/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/A461CC43-03ED-DE11-8E44-00304867BFF2.root'

    )
    , duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

)

process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('124120:1-124120:59')

# this is for filtering on L1 technical trigger bit
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND ( 40 OR 41 )')

#### remove beam scraping events
process.noScraping= cms.EDFilter("FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.20)
)

#process.dqmBeamMonitor.Debug = True
#process.dqmBeamMonitor.BeamFitter.Debug = True
process.dqmBeamMonitor.BeamFitter.WriteAscii = True
process.dqmBeamMonitor.BeamFitter.AsciiFileName = 'BeamFitResults.txt'
#process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
process.dqmBeamMonitor.BeamFitter.DIPFileName = 'BeamFitResults.txt'
#process.dqmBeamMonitor.BeamFitter.SaveFitResults = True
process.dqmBeamMonitor.BeamFitter.OutputFileName = 'BeamFitResults.root'
#process.dqmBeamMonitor.resetEveryNLumi = 10
#process.dqmBeamMonitor.resetPVEveryNLumi = 5

## bx
#process.dqmBeamMonitorBx.Debug = True
#process.dqmBeamMonitorBx.BeamFitter.Debug = True
process.dqmBeamMonitorBx.BeamFitter.WriteAscii = True
process.dqmBeamMonitorBx.BeamFitter.AsciiFileName = 'BeamFitResultsBx.txt'

### TKStatus
process.dqmTKStatus = cms.EDAnalyzer("TKStatus",
	BeamFitter = cms.PSet(
	DIPFileName = process.dqmBeamMonitor.BeamFitter.DIPFileName
	)
)
###

process.pp = cms.Path(process.dqmTKStatus*process.hltLevel1GTSeed*process.dqmBeamMonitor+process.dqmBeamMonitorBx+process.dqmEnv+process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = 'cmslpc17.fnal.gov'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True


process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(cms.PSet(
    record = cms.string('BeamSpotObjectsRcd'),
    tag = cms.string('Early10TeVCollision_3p8cm_v3_mc_IDEAL')
    )),
#    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT')
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_BEAMSPOT')
)

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.schedule = cms.Schedule(process.pp)

