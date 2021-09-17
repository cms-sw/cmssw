import FWCore.ParameterSet.Config as cms

process = cms.Process('DQMCMMonitor')

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')

process.source = cms.Source(
  "PoolSource",
  fileNames = cms.untracked.vstring(
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning08/Run69750_FEED31F3-58AC-DD11-BF73-000423D99658.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning08/Run69800_026DBE87-A5AC-DD11-9397-0030487C608C.root'
        #'file:/home/magnan/SOFTWARE/CMS/CMSSW_3_1_0_pre11/src/FedWorkDir/FedMonitoring/test/Digi_run69800.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning08/Run69797_FC26431D-91AC-DD11-A0D1-001617E30CC8.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning08/Run69874_98BB9120-E6AC-DD11-9B91-000423D99896.root'
        'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning09/Run106019_00D9F347-4D72-DE11-93F6-001D09F24399.root'
        #'file:/home/magnan/SOFTWARE/CMS/data/FED/Commissioning09/Run101045_A6F7D0D3-4560-DE11-A52A-001D09F2545B.root'
        ),
  skipBadFiles = cms.untracked.bool(True),                        
  #inputCommands = cms.untracked.vstring('drop *', 'keep *_source_*_*'),

  )

#process.load("DQM.SiStripMonitorHardware.test.source_cff")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )

#process.service = cms.ProfilerService {
#    untracked int32 firstEvent = 1
#    untracked int32 lastEvent = 50
#    untracked vstring paths = { "p"}
#    }

#process.load('DQM.SiStripCommon.MessageLogger_cfi')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        noLineBreaks = cms.untracked.bool(False),
        threshold = cms.untracked.string('ERROR')
    ),
    files = cms.untracked.PSet(
        debug = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('DEBUG')
        ),
        error = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('ERROR')
        ),
        info = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('INFO')
        ),
        warning = cms.untracked.PSet(
            noLineBreaks = cms.untracked.bool(False),
            threshold = cms.untracked.string('WARNING')
        )
    ),
    suppressDebug = cms.untracked.vstring(),
    suppressInfo = cms.untracked.vstring(),
    suppressWarning = cms.untracked.vstring()
)


#needed to produce tkHistoMap
process.load("DQM.SiStripCommon.TkHistoMap_cff")

# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "GR09_31X_V1P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
# Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")


process.load("DPGAnalysis.SiStripTools.apvshotsanalyzer_cfi")

process.load('DQM.SiStripMonitorHardware.siStripCMMonitor_cfi')
process.siStripCMMonitor.FillWithEventNumber = False
process.siStripCMMonitor.FillWithLocalEventNumber = False
process.siStripCMMonitor.FedIdVec = 100,200,400
process.siStripCMMonitor.PrintDebugMessages = 1
process.siStripCMMonitor.WriteDQMStore = True
process.siStripCMMonitor.DQMStoreFileName = "DQMStore_CM_run106019.root"

#process.siStripCMMonitor.TimeHistogramConfig.NBins = 100
#process.siStripCMMonitor.TimeHistogramConfig.Min = 0
#process.siStripCMMonitor.TimeHistogramConfig.Max = 1

process.load('PerfTools.Callgrind.callgrindSwitch_cff')

process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string("Shot_run106019.root"),
                                   closeFileFast = cms.untracked.bool(True)
                                   )


process.p = cms.Path( #process.profilerStart*
                      process.siStripDigis
                      *process.siStripZeroSuppression
                      *process.apvshotsanalyzer
                      *process.siStripCMMonitor
                      #*process.profilerStop 
                      )

process.saveDigis = cms.OutputModule( 
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *_*_*_HLT',
        'drop *_*_*Raw_DQMCMMonitor',
        'drop *_*_ScopeMode_DQMCMMonitor',
        'keep *_siStripDigis_ZeroSuppressed_*',
        'keep *_source_*_*'
        ),
    fileName = cms.untracked.string('Digi_run106019.root')
    )

process.pout = cms.EndPath( process.saveDigis )
