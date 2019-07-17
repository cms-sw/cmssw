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
    input = cms.untracked.int32(2)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/NoPU_90X_upgrade2023_realistic_v9-v1/70000/44C2F01A-DE26-E711-A085-FA163E0162D6.root',
        '/store/mc/PhaseIISpring17D/SingleE_FlatPt-8to100/GEN-SIM-DIGI-RAW/NoPU_90X_upgrade2023_realistic_v9-v1/70000/70829AD5-1526-E711-B695-FA163E5613EB.root'
    ),
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_9_1_0_pre3/RelValSingleElectronPt10Extended/GEN-SIM-DIGI-RAW/91X_upgrade2023_realistic_v1_D11-v1/10000/10597704-722E-E711-A2CD-0CC47A78A436.root'),
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

process.load("L1Trigger.L1TTrackMatch.L1TkObjectProducers_cff")
process.pL1TkObjects = cms.Path(process.L1TkElectrons
                             + process.L1TkPhotons
                             + process.L1TkCaloJets
                             + process.L1TkPrimaryVertex
                             + process.L1TrackerEtMiss
                             + process.L1TkCaloHTMissVtx
                             + process.L1TkMuons
                             + process.L1TkTauFromCalo)

process.printTkObj = cms.EDAnalyzer( 'PrintL1TkObjects' ,
    L1TkVtxInputTag       = cms.InputTag("L1TkPrimaryVertex",""),
    L1TkEtMissInputTag    = cms.InputTag("L1TrackerEtMiss","MET"),
    L1TkJetsInputTag      = cms.InputTag("L1TkCaloJets","L1TkCaloJets"),
    L1TkHTMInputTag       = cms.InputTag("L1TkCaloHTMissVtx", "L1TkCaloHTMiss"),
    L1TkPhotonsInputTag   = cms.InputTag("L1TkPhotons" "EG"),
    L1TkElectronsInputTag = cms.InputTag("L1TkElectrons","EG"),
    L1TkMuonsInputTag     = cms.InputTag("L1TkMuons",""),
    L1TkTausInputTag      = cms.InputTag("L1TkTauFromCalo")

)
process.pPrintObject = cms.Path(process.printTkObj)
# root file with histograms produced by the analyzer

# ---------------------------------------------------------------------------

process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.pDumpED = cms.Path(process.dumpED)


process.schedule = cms.Schedule(process.L1simulation_step,process.L1TrackTrigger_step,process.pL1TkObjects,process.pPrintObject,process.pDumpED)
#process.schedule = cms.Schedule(process.L1simulation_step,process.L1TrackTrigger_step)
