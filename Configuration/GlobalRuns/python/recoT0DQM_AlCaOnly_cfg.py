
import FWCore.ParameterSet.Config as cms

process = cms.Process("Rec")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")


process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(1000) )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/006945C8-40A5-DD11-BD7E-001617DBD556.root'
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
    version = cms.untracked.string('$Revision: 1.2 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/GlobalRuns/python/recoT0DQM_AlCaOnly_cfg.py,v $'),
    annotation = cms.untracked.string('CRUZET Prompt Reco with DQM with Mag field at 3.8T.  Only AlCaReco is output')
)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) ## default is false


# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "CRAFT_ALL_V3::All"
process.prefer("GlobalTag")

# Magnetic fiuld: force mag field to be 3.8 tesla
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

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
process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")
process.load("L1Trigger.Configuration.L1Config_cff")
#process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFConfigProducer_cfi")
#process.load("L1TriggerConfig.CSCTFConfigProducers.L1MuCSCTFConfigurationRcdSrc_cfi")

#Paths
process.raw2digi_step = cms.Path(process.RawToDigi)
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
