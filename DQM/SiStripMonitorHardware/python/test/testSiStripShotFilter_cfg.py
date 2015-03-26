import FWCore.ParameterSet.Config as cms

process = cms.Process('test')

process.source = cms.Source(
  "PoolSource",
  fileNames = cms.untracked.vstring(
        '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/121/632/00BE1303-F3D4-DE11-AA08-001D09F29169.root'
        ),
  skipBadFiles = cms.untracked.bool(True),                        
  #inputCommands = cms.untracked.vstring('drop *', 'keep *_source_*_*'),

  )

#process.load("DQM.SiStripMonitorHardware.test.source_cff")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )

# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "GR09_P_V6::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
# Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

##########################################################
#needed to produce tkHistoMap+shot histos.
#Can comment if commenting apvshotsanalyzer in main path
process.DQMStore = cms.Service("DQMStore")
process.TkDetMap = cms.Service("TkDetMap")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
process.load("DPGAnalysis.SiStripTools.apvshotsanalyzer_cfi")
process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string("Shot.root"),
                                   closeFileFast = cms.untracked.bool(True)
                                   )
###########################################################

process.load('DQM.SiStripMonitorHardware.siStripShotFilter_cfi')
process.siStripShotFilter.OutputFilePath = "shotChannels.dat"

process.p = cms.Path( 
    process.siStripDigis
    *process.siStripZeroSuppression
    *process.apvshotsanalyzer
    *process.siStripShotFilter
    )

#process.saveDigis = cms.OutputModule( 
#    "PoolOutputModule",
#    outputCommands = cms.untracked.vstring(
#        'drop *_*_*_HLT',
#        'drop *_*_*Raw_DQMCMMonitor',
#        'drop *_*_ScopeMode_DQMCMMonitor',
#        'keep *_siStripDigis_ZeroSuppressed_*',
#        'keep *_source_*_*'
#        ),
#    fileName = cms.untracked.string('Digi_run106019.root')
#    )

#process.pout = cms.EndPath( process.saveDigis )
