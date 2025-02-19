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

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/data1/heavyion/0662E9D1-D070-DF11-8329-001E68865F6D.root',
        'file:/data1/heavyion/06B2AC18-8170-DF11-A5E0-001E6878F716.root',
        'file:/data1/heavyion/0A61729E-8270-DF11-A629-00E081300BDA.root',
        'file:/data1/heavyion/0C0A25E3-8070-DF11-BF64-001E688629B3.root',
        'file:/data1/heavyion/246A161A-8170-DF11-A8B1-001E6849D384.root',
        'file:/data1/heavyion/3679ED40-8070-DF11-8A80-001E6865A59A.root',
        'file:/data1/heavyion/409BE308-BD71-DF11-8494-00188B7ABC14.root',
        'file:/data1/heavyion/445EF2FD-7F70-DF11-BF0A-00188B7ABC0E.root',
        'file:/data1/heavyion/4CB4830C-8270-DF11-B4A7-001E6849D21C.root',
        'file:/data1/heavyion/54A93027-3A71-DF11-8B43-00E081300BDA.root',
        'file:/data1/heavyion/580BCF18-8170-DF11-B39A-001E6878F716.root',
        'file:/data1/heavyion/5AA1030A-8270-DF11-9DB8-00188B7AC862.root',
        'file:/data1/heavyion/5E068F91-9770-DF11-92D9-001E6878FA76.root',
        'file:/data1/heavyion/5E5DF809-8270-DF11-86F2-00188B7ACD60.root',
        'file:/data1/heavyion/72A5783E-8070-DF11-ACBC-00E08130DD28.root',
        'file:/data1/heavyion/865B5019-8170-DF11-ADF2-001E6878FB3A.root',
        'file:/data1/heavyion/8EA97318-8170-DF11-8C7E-00E0813006F4.root',
        'file:/data1/heavyion/9086C3E5-7F70-DF11-94B7-0026B95C2418.root',
        'file:/data1/heavyion/AEA8BE40-8070-DF11-B89F-0026B95BCAC3.root',
        'file:/data1/heavyion/B48E08CF-7F70-DF11-BDEE-00E081300BDA.root',
        'file:/data1/heavyion/B69BA9EF-8170-DF11-9D35-001E68659F36.root',
        'file:/data1/heavyion/C2F8E48F-9770-DF11-97F0-00E0813000C2.root',
        'file:/data1/heavyion/EACFBC1A-8170-DF11-996D-001E68865F71.root',
        'file:/data1/heavyion/FA3B59E5-7F70-DF11-BC81-001E68862AE3.root',
        'file:/data1/heavyion/FC6349FD-7F70-DF11-AE96-00188B7ACD5D.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0005/54A93027-3A71-DF11-8B43-00E081300BDA.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0005/409BE308-BD71-DF11-8494-00188B7ABC14.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0005/0662E9D1-D070-DF11-8329-001E68865F6D.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/FC6349FD-7F70-DF11-AE96-00188B7ACD5D.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/FA3B59E5-7F70-DF11-BC81-001E68862AE3.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/EACFBC1A-8170-DF11-996D-001E68865F71.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/C2F8E48F-9770-DF11-97F0-00E0813000C2.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/B69BA9EF-8170-DF11-9D35-001E68659F36.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/B48E08CF-7F70-DF11-BDEE-00E081300BDA.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/AEA8BE40-8070-DF11-B89F-0026B95BCAC3.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/9086C3E5-7F70-DF11-94B7-0026B95C2418.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/8EA97318-8170-DF11-8C7E-00E0813006F4.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/865B5019-8170-DF11-ADF2-001E6878FB3A.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/72A5783E-8070-DF11-ACBC-00E08130DD28.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/5E5DF809-8270-DF11-86F2-00188B7ACD60.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/5E068F91-9770-DF11-92D9-001E6878FA76.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/5AA1030A-8270-DF11-9DB8-00188B7AC862.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/580BCF18-8170-DF11-B39A-001E6878F716.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/4CB4830C-8270-DF11-B4A7-001E6849D21C.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/445EF2FD-7F70-DF11-BF0A-00188B7ABC0E.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/3679ED40-8070-DF11-8A80-001E6865A59A.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/246A161A-8170-DF11-A8B1-001E6849D384.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/0C0A25E3-8070-DF11-BF64-001E688629B3.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/0A61729E-8270-DF11-A629-00E081300BDA.root',
        #'/store/mc/Spring10/Hydjet_Quenched_MinBias_2760GeV/GEN-SIM-RECO/MC_36Y_V7A-v1/0002/06B2AC18-8170-DF11-A5E0-001E6878F716.root',
    )
    , duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

)

#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('124120:1-124120:59')

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

process.pp = cms.Path(process.dqmTKStatus*
                      #process.hltLevel1GTSeed*
                      process.dqmBeamMonitor+
                      #process.dqmBeamMonitorBx+
                      process.dqmEnv+
                      process.dqmSaver)

process.DQMStore.verbose = 0
#process.DQM.collectorHost = 'cmslpc17.fnal.gov'
process.DQM.collectorHost = 'localhost'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_36Y_V7A::All'

#process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        #process.CondDBSetup,
                                        #toGet = cms.VPSet(cms.PSet(
    #record = cms.string('BeamSpotObjectsRcd'),
    #tag = cms.string('Early10TeVCollision_3p8cm_v3_mc_IDEAL')
    #)),
##    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT')
    #connect = cms.string('frontier://FrontierProd/CMS_COND_31X_BEAMSPOT')
#)

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.schedule = cms.Schedule(process.pp)

