import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RERUNL1',eras.Phase2C2_timing)

process.load("FWCore.MessageService.MessageLogger_cfi")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('SimCalorimetry.HcalTrigPrimProducers.hcalTTPDigis_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#       '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/NoPU_90X_upgrade2023_realistic_v9-v1/70000/44C2F01A-DE26-E711-A085-FA163E0162D6.root'
#       '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/NoPU_90X_upgrade2023_realistic_v9-v1/70000/70829AD5-1526-E711-B695-FA163E5613EB.root'
#  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/1010000/067329A9-F42B-E711-9A8B-7845C4F91450.root'
#  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/028C566F-142C-E711-8D91-008CFAF74A86.root'
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/06D64EE3-132C-E711-A393-7845C4FC3B57.root',
#  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/10B73E35-142C-E711-AD41-F04DA275BFDD.root',
#  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/2E55A586-132C-E711-9754-008CFAF73424.root',
#  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/36A41F6C-162C-E711-AE4B-848F69FD2484.root',
#  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/38133A74-142C-E711-8373-7845C4F92F7B.root'
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/3AF42BDD-122C-E711-AC80-008CFAFBFB94.root',
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/3E27D4D9-132C-E711-A5A9-848F69FD4667.root',
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/3E90E8E3-132C-E711-8113-7845C4FC3B57.root',
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/42C02741-162C-E711-92B3-008CFAF3543C.root',
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/4496B492-142C-E711-9694-008CFAF73286.root',
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/480790E6-152C-E711-AD1E-008CFAFBEBF8.root',
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/4A1C3D50-152C-E711-B07F-848F69FD2997.root',
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/58CF6805-142C-E711-B043-008CFAF721CA.root',
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/6CD231C8-162C-E711-8BFB-008CFAF73658.root',
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/78DFD21E-152C-E711-B84C-008CFAF74750.root',
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/7E8AF76F-162C-E711-B522-008CFAF35AC0.root',
##  '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/120000/966971C8-152C-E711-94A1-7845C4F91495.root',
'/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/70000/004AB49D-1326-E711-9A18-001E67792702.root'
##'/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/70000/004BF05D-2526-E711-BB9E-001E67792768.root',
##'/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/70000/0051D410-0B26-E711-BD33-001E677928DA.root',
##'/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/70000/005C4D4B-D925-E711-B46B-0025905C431C.root',
##'/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/70000/00761729-3026-E711-9B1C-001E677928C6.root',
##'/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/70000/00D388E4-0F26-E711-8F52-0025901D49AC.root',
##'/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/70000/0222BF5F-2626-E711-9737-A4BF0108B062.root',
##'/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/70000/02249200-0B26-E711-A2B1-001E6779241A.root',
##'/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/PU140_90X_upgrade2023_realistic_v9-v1/70000/0253CC61-1326-E711-8897-001E6739713E.root'
    ),
    secondaryFileNames = cms.untracked.vstring()
)
# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Additional output definition

#process.Timing = cms.Service("Timing")


# Other statements
process.mix.input.nbPileupEvents.averageNumber = cms.double(200.000000)
process.mix.bunchspace = cms.int32(25)
process.mix.minBunch = cms.int32(-12)
process.mix.maxBunch = cms.int32(3)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '90X_upgrade2023_realistic_v9', '')


process.HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP",
    UseOldLoader = cms.bool(False)
)


process.HcalTPGCoderULUT = cms.ESProducer("HcalTPGCoderULUT",
    FGLUTs = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/HBHE_FG_LUT.dat'),
    LUTGenerationMode = cms.bool(True),
    MaskBit = cms.int32(32768),
    RCalibFile = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/RecHit-TPG-calib.dat'),
    inputLUTs = cms.FileInPath('CalibCalorimetry/HcalTPGAlgos/data/inputLUTcoder_physics.dat'),
    read_Ascii_LUTs = cms.bool(False),
    read_FG_LUTs = cms.bool(False),
    read_XML_LUTs = cms.bool(False)
)


process.HcalTrigTowerGeometryESProducer = cms.ESProducer("HcalTrigTowerGeometryESProducer")

process.CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL', 
        'ZDC', 
        'EcalBarrel', 
        'TOWER', 
        'HGCalEESensitive', 
        'HGCalHESiliconSensitive')
)


process.CaloTPGTranscoder = cms.ESProducer("CaloTPGTranscoderULUTs",
    HFTPScaleShift = cms.PSet(
        NCT = cms.int32(1),
        RCT = cms.int32(3)
    ),
    LUTfactor = cms.vint32(1, 2, 5, 0),
    RCTLSB = cms.double(0.25),
    ZS = cms.vint32(4, 2, 1, 0),
    hcalLUT1 = cms.FileInPath('CalibCalorimetry/CaloTPG/data/outputLUTtranscoder_physics.dat'),
    hcalLUT2 = cms.FileInPath('CalibCalorimetry/CaloTPG/data/TPGcalcDecompress2.txt'),
    ietaLowerBound = cms.vint32(1, 18, 27, 29),
    ietaUpperBound = cms.vint32(17, 26, 28, 32),
    nominal_gain = cms.double(0.177),
    read_Ascii_Compression_LUTs = cms.bool(False),
    read_Ascii_RCT_LUTs = cms.bool(False)
)


process.CaloTopologyBuilder = cms.ESProducer("CaloTopologyBuilder")


process.CaloTowerHardcodeGeometryEP = cms.ESProducer("CaloTowerHardcodeGeometryEP")


process.CaloTowerTopologyEP = cms.ESProducer("CaloTowerTopologyEP")

#from SimTracker.TrackTriggerAssociation.TTTrackAssociation_cfi import *
#TrackTriggerAssociatorTracks = cms.Sequence(TTTrackAssociatorFromPixelDigis)

process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')
# Remove best choice selection
process.hgcalTriggerPrimitiveDigiProducer.FECodec.NData = cms.uint32(999)
process.hgcalTriggerPrimitiveDigiProducer.FECodec.DataLength = cms.uint32(8)
process.hgcalTriggerPrimitiveDigiProducer.FECodec.triggerCellTruncationBits = cms.uint32(7)

process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].calib_parameters.cellLSB = cms.double(
        process.hgcalTriggerPrimitiveDigiProducer.FECodec.linLSB.value() * 
        2 ** process.hgcalTriggerPrimitiveDigiProducer.FECodec.triggerCellTruncationBits.value() 
)

cluster_algo_all =  cms.PSet( AlgorithmName = cms.string('HGCClusterAlgoBestChoice'),
                              FECodec = process.hgcalTriggerPrimitiveDigiProducer.FECodec,
                              HGCalEESensitive_tag = cms.string('HGCalEESensitive'),
                              HGCalHESiliconSensitive_tag = cms.string('HGCalHESiliconSensitive'),                           
                              calib_parameters = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].calib_parameters,
                              C2d_parameters = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C2d_parameters,
                              C3d_parameters = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C3d_parameters
                              )


process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms = cms.VPSet( cluster_algo_all )

process.hgcl1tpg_step = cms.Path(process.hgcalTriggerPrimitives)


# load ntuplizer
#process.load('L1Trigger.L1THGCal.hgcalTriggerNtuples_cff')
#process.ntuple_step = cms.Path(process.hgcalTriggerNtuples)

process.load('SimCalorimetry.EcalEBTrigPrimProducers.ecalEBTriggerPrimitiveDigis_cff')
process.EcalEBtp_step = cms.Path(process.simEcalEBTriggerPrimitiveDigis)

# Path and EndPath definitions
process.HcalTPsimulation_step = cms.Path(process.hcalTTPSequence)
process.L1simulation_step = cms.Path(process.SimL1Emulator)

process.load('L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff')
process.L1TrackTrigger_step = cms.Path(process.L1TrackletTracks)

process.endjob_step = cms.EndPath(process.endOfProcess)
# ---------------------------------------------------------------------------

# Now we produce L1TkEmParticles and L1TkElectrons


# ----  "electrons" from L1Tracks. Inclusive electrons :

process.load("L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi")
process.L1TkElectrons.L1TrackInputTag = cms.InputTag("TTTracksFromTracklet","Level1TTTracks" )
process.L1TkIsoElectrons.L1TrackInputTag = cms.InputTag("TTTracksFromTracklet","Level1TTTracks" )

process.pElectrons = cms.Path( process.L1TkElectrons + process.L1TkIsoElectrons)

process.L1TrkAna = cms.EDAnalyzer( 'L1TkElectronAnalyzer' ,
    L1EGammaInputTag = cms.InputTag("simCaloStage2Digis",""),
    L1TkElectronInputTag = cms.InputTag("L1TkElectrons","EG"),
    L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
    GenParticleInputTag = cms.InputTag("genParticles",""),
    AnalysisOption   = cms.string("Efficiency"),
    EtaCutOff   = cms.double(2.5),
    TrackPtCutOff   = cms.double(10.0),
    GenPtThreshold   = cms.double(20.0),
    EGammaEtThreshold = cms.double(20.0)                              
)

process.L1IsoTrkAna = process.L1TrkAna.clone()
process.L1IsoTrkAna.L1TkElectronInputTag = cms.InputTag("L1TkIsoElectrons","EG")

# root file with histograms produced by the analyzer
filename = "efficiency_PU140_1.root"
process.TFileService = cms.Service("TFileService", fileName = cms.string(filename), closeFileFast = cms.untracked.bool(True))

process.pTrkAna = cms.Path( process.L1TrkAna + process.L1IsoTrkAna)
# ---------------------------------------------------------------------------

process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.pDumpED = cms.Path(process.dumpED)


process.schedule = cms.Schedule(process.L1simulation_step,process.L1TrackTrigger_step,process.pElectrons,process.pTrkAna)








