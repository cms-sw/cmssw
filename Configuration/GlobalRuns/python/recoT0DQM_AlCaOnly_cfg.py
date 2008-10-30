
import FWCore.ParameterSet.Config as cms

process = cms.Process("Rec")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")


process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(1000) )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
     '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/375/FC48E9B6-DC9A-DD11-AEEA-000423D6BA18.root'
    )
)

# AlCaReco output modules
process.ALCARECOStreamRpcCalHLT = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECORpcCalHLT')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_muonDTDigis_*_*', 
        'keep CSCDetIdCSCWireDigiMuonDigiCollection_*_*_*', 
        'keep CSCDetIdCSCStripDigiMuonDigiCollection_*_*_*', 
        'keep DTLayerIdDTDigiMuonDigiCollection_*_*_*', 
        'keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*', 
        'keep L1MuGMTCands_*_*_*', 
        'keep L1MuGMTReadoutCollection_*_*_*'),
    fileName = cms.untracked.string('ALCARECORpcCalHLT.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECORpcCalHLT'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)
process.ALCARECOStreamMuAlCalIsolatedMu = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlCalIsolatedMu')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_ALCARECOMuAlCalIsolatedMu_*_*', 
        'keep *_muonCSCDigis_*_*', 
        'keep *_muonDTDigis_*_*', 
        'keep *_muonRPCDigis_*_*', 
        'keep *_dt1DRecHits_*_*', 
        'keep *_dt2DSegments_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*'),
    fileName = cms.untracked.string('ALCARECOMuAlCalIsolatedMu.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('StreamALCARECOMuAlCalIsolatedMu'),
        dataTier = cms.untracked.string('ALCARECO')
    )
)

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/GlobalRuns/python/recoT0DQM_AlCaOnly_cfg.py,v $'),
    annotation = cms.untracked.string('CRUZET Prompt Reco with DQM with Mag field at 0T.  Only AlCaReco is output')
)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) ## default is false


# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V1P::All"
process.prefer("GlobalTag")

# Magnetic fiuld: force mag field to be 0 tesla
process.load("Configuration.StandardSequences.MagneticField_0T_cff")

#Geometry
process.load("Configuration.StandardSequences.Geometry_cff")

# Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

# reconstruction sequence for Cosmics
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

# AlCaReco
process.load('Configuration/StandardSequences/AlCaRecoStreams_cff')

# offline DQM
process.load("DQMOffline.Configuration.DQMOfflineCosmics_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cff")

#L1 trigger validation
#process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")
process.load("L1Trigger.Configuration.L1Config_cff")
process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFConfigProducer_cfi")
process.load("L1TriggerConfig.CSCTFConfigProducers.L1MuCSCTFConfigurationRcdSrc_cfi")

process.roadSearchSeedsP5.MaxNumberOfCosmicClusters = 100

#Paths
process.raw2digi_step = cms.Path(process.RawToDigi_woGCT)
process.reconstruction_step = cms.Path(process.reconstructionCosmics)
process.dqm_step = cms.Path(process.DQMOfflineCosmics * process.MEtoEDMConverter)
process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu)
process.ALCARECOStreamRpcCalHLTOutPath = cms.EndPath(process.ALCARECOStreamRpcCalHLT)
process.ALCARECOStreamMuAlCalIsolatedMuOutPath = cms.EndPath(process.ALCARECOStreamMuAlCalIsolatedMu)

process.ALCARECOMuAlCalIsolatedMuHLT.HLTPaths = ['HLT_L1MuOpen']
process.ALCARECORpcCalHLTFilter.HLTPaths = ['HLT_L1MuOpen']

# Schedule definition
process.schedule = cms.Schedule( process.raw2digi_step,process.reconstruction_step,process.pathALCARECORpcCalHLT,process.pathALCARECOMuAlCalIsolatedMu,process.dqm_step,process.ALCARECOStreamRpcCalHLTOutPath,process.ALCARECOStreamMuAlCalIsolatedMuOutPath)
