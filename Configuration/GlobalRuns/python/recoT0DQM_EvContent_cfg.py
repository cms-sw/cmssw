
import FWCore.ParameterSet.Config as cms

process = cms.Process("Rec")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")


process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource",
    #firstFreeID = cms.untracked.uint32(131072),
    fileNames = cms.untracked.vstring(
    '/store/temp/data/Commissioning08/Cosmics/RAW/CRUZET4_v1/000/057/289/0851D62F-096D-DD11-92F3-000423D98DB4.root'
 #/store/data/Commissioning08/Cosmics/RAW/MW33_v1/000/056/520/467B9D51-4C6A-DD11-BD56-000423D98844.root'      
 # '/store/data/CRUZET3/Cosmics/RAW/v4/000/050/908/04D61810-DD54-DD11-8FB1-001617DBD230.root'
       #'/store/data/CRUZET3/Cosmics/RAW/v4/000/050/908/08CDE58E-DC54-DD11-9D2A-000423D998BA.root',
       #'/store/data/CRUZET3/Cosmics/RAW/v4/000/050/908/0EDB772A-DC54-DD11-8C5C-001617C3B77C.root',
             #'/store/data/Commissioning08/Cosmics/RAW/MW31_v1/000/053/416/68D08BC6-555E-DD11-842C-000423D9939C.root'
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
    version = cms.untracked.string('$Revision: 1.10 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/GlobalRuns/python/recoT0DQM_EvContent_cfg.py,v $'),
    annotation = cms.untracked.string('CRUZET Prompt Reco with DQM')
)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) ## default is false


# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V2P::All"
process.prefer("GlobalTag")

# Magnetic fiuld: force mag field to be 0.0 tesla
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

#Geometry
process.load("Configuration.StandardSequences.Geometry_cff")

# Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

# reconstruction sequence for Cosmics
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

# patch needed for CRUZET2 but not used in CRUZET3 cfg
process.SteppingHelixPropagatorAny.useInTeslaFromMagField = True
process.SteppingHelixPropagatorAlong.useInTeslaFromMagField = True
process.SteppingHelixPropagatorOpposite.useInTeslaFromMagField = True
process.SteppingHelixPropagatorAny.SetVBFPointer = True
process.SteppingHelixPropagatorAlong.SetVBFPointer = True
process.SteppingHelixPropagatorOpposite.SetVBFPointer = True
process.VolumeBasedMagneticFieldESProducer.label = 'VolumeBasedMagneticField'


# offline DQM
process.load("DQMOffline.Configuration.DQMOffline_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cff")

#L1 trigger validation
#process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")
process.load("L1Trigger.Configuration.L1Config_cff")
process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFConfigProducer_cfi")
process.load("L1TriggerConfig.CSCTFConfigProducers.L1MuCSCTFConfigurationRcdSrc_cfi")

#
## add quality info
#
process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.SiStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
     cms.PSet( record = cms.string("SiStripFedCablingRcd"), tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadChannelRcd"), tag    = cms.string("") ),
     cms.PSet( record = cms.string("SiStripBadFiberRcd"),   tag    = cms.string("") )
)
process.prefer("SiStripQualityESProducer")

# ad hoc changes for V3 reco to lower seed threshold to 5 GeV
process.cosmicseedfinderP5.SeedPt = 5
process.combinatorialcosmicseedfinderP5.SeedMomentum = 5

process.load("DQMOffline.Muon.muonCosmicMonitors_cff")

process.allPath = cms.Path( process.RawToDigi_woGCT * process.reconstructionCosmics *  process.DQMOffline * process.MEtoEDMConverter)
#process.allPath_noDQM = cms.Path( process.RawToDigi_woGCT * process.reconstructionCosmics )

process.outpath = cms.EndPath(process.FEVT)
