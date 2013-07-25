
import FWCore.ParameterSet.Config as cms

process = cms.Process("Rec")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")


process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(20) )
#can be 300 in that file
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#MC relval SINGLE MUON
    ###'/store/relval/CMSSW_2_1_9/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0000/EA123BEB-9F85-DD11-AB1D-000423D9870C.root'

#MC relval singlemu
    #'/store/relval/CMSSW_3_1_0_pre3/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/581655A2-7D0A-DE11-B548-0019DB2F3F9A.root'

#MC relval ttbar
    '/store/relval/CMSSW_3_1_0_pre3/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/1EE2ECB6-170A-DE11-BE8C-0016177CA7A0.root',  
    #'/store/relval/CMSSW_3_1_0_pre3/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0001/3C8AABDF-FA0A-DE11-80A5-001D09F290BF.root'
    )
)

# output module
#
process.load("Configuration.EventContent.EventContentCosmics_cff")

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RECO')),
    fileName = cms.untracked.string('promptrecoCosmics.root')
)

process.FEVT.outputCommands.append('keep CaloTowersSorted_calotoweroptmaker_*_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCALCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCALCTDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCCLCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCCLCTDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCComparatorDigiMuonDigiCollection_muonCSCDigis_MuonCSCComparatorDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCCorrelatedLCTDigiMuonDigiCollection_csctfDigis_*_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCCorrelatedLCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCCorrelatedLCTDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCRPCDigiMuonDigiCollection_muonCSCDigis_MuonCSCRPCDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCStripDigiMuonDigiCollection_muonCSCDigis_MuonCSCStripDigi_*')
process.FEVT.outputCommands.append('keep CSCDetIdCSCWireDigiMuonDigiCollection_muonCSCDigis_MuonCSCWireDigi_*')
process.FEVT.outputCommands.append('keep cscL1TrackCSCDetIdCSCCorrelatedLCTDigiMuonDigiCollectionstdpairs_csctfDigis_*_*')
process.FEVT.outputCommands.append('keep DTChamberIdDTLocalTriggerMuonDigiCollection_muonDTDigis_*_*')
process.FEVT.outputCommands.append('keep DTLayerIdDTDigiMuonDigiCollection_muonDTDigis_*_*')
process.FEVT.outputCommands.append('keep intL1CSCSPStatusDigisstdpair_csctfDigis_*_*')
process.FEVT.outputCommands.append('keep L1MuDTChambPhContainer_dttfDigis_*_*')
process.FEVT.outputCommands.append('keep L1MuDTChambThContainer_dttfDigis_*_*')
process.FEVT.outputCommands.append('keep L1MuDTTrackContainer_dttfDigis_DATA_*')
process.FEVT.outputCommands.append('keep PixelDigiedmDetSetVector_siPixelDigis_*_*')
process.FEVT.outputCommands.append('keep recoCandidatesOwned_caloTowersOpt_*_*')
process.FEVT.outputCommands.append('keep RPCDetIdRPCDigiMuonDigiCollection_muonRPCDigis_*_*')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DQM/SiPixelMonitorTrack/test/SiPixelMonitorTrackResiduals_mc_cfg.py,v $'),
    annotation = cms.untracked.string('CRUZET Prompt Reco with DQM with Mag field at 0T')
)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) ## default is false

# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
####process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "IDEAL_V9::All"
process.GlobalTag.globaltag = "IDEAL_30X::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')
###process.prefer("GlobalTag")

# Magnetic field: force mag field to be 0 tesla
process.load("Configuration.StandardSequences.MagneticField_0T_cff")

#Geometry
process.load("Configuration.StandardSequences.Geometry_cff")

# Real data raw to digi
##process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")

# reconstruction sequence for Cosmics
#process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

# offline DQM
##process.load("DQMOffline.Configuration.DQMOfflineCosmics_cff")
process.load("DQMOffline.Configuration.DQMOffline_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cff")

#L1 trigger validation
#process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")
process.load("L1Trigger.Configuration.L1Config_cff")
process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFConfigProducer_cfi")
process.load("L1TriggerConfig.CSCTFConfigProducers.L1MuCSCTFConfigurationRcdSrc_cfi")

##new##
process.load("Configuration.StandardSequences.Simulation_cff")

#MC
process.SiPixelTrackResidualSource.TrackCandidateProducer = cms.string('newTrackCandidateMaker')
process.SiPixelTrackResidualSource.trajectoryInput = cms.InputTag('generalTracks')
process.SiPixelTrackResidualSource.saveFile = cms.untracked.bool(True)
process.SiPixelTrackResidualSource.modOn = cms.untracked.bool(True)
process.SiPixelClusterSource.modOn = cms.untracked.bool(True)

#event content analyzer
process.dump = cms.EDAnalyzer('EventContentAnalyzer')

#Paths
##process.allPath = cms.Path( process.RawToDigi_woGCT * process.reconstructionCosmics *  process.DQMOfflineCosmics * process.MEtoEDMConverter)
##process.allPath = cms.Path( process.RawToDigi * process.reconstruction * process.DQMOffline * process.MEtoEDMConverter)

#######process.allPath = cms.Path( process.RawToDigi * process.reconstruction * process.DQMOffline)

process.allPath = cms.Path( process.RawToDigi * process.reconstruction * process.SiPixelClusterSource * process.SiPixelTrackResidualSource)

process.outpath = cms.EndPath(process.FEVT)
