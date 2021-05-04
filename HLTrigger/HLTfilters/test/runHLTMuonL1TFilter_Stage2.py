# Example of simple HLT filter selecting Stage2 L1 muons
# Stage2: Unpackers + GT Emulator + HLT seeding + HLT filter
#
# V.Rekovic

import FWCore.ParameterSet.Config as cms


#from Configuration.Eras.Era_Run2_25ns_cff import Run2_25ns
#process = cms.Process('L1SEQS',Run2_25ns)
from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
process = cms.Process('HLT',Run2_2016)


# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring(
        'hltL1sL1SingleMuBeamHalo', 
        'hltL1fL1SingleMuBeamHaloFiltered0'
    ),
    files = cms.untracked.PSet(
        critical = cms.untracked.PSet(

        ),
        detailedInfo = cms.untracked.PSet(
            threshold = cms.untracked.string('DEBUG')
        )
    )
)

#
# LOCAL CONDITIONS NEEDED FOR RE-EMULATION OF GT
#

from L1Trigger.L1TGlobal.GlobalParameters_cff import *
from L1Trigger.L1TGlobal.TriggerMenu_cff import *
TriggerMenu.L1TriggerMenuFile = cms.string('L1Menu_Collisions2015_25nsStage1_v7_uGT.xml')


# ####################################################
# BEGIN L1T UNPACKER-EMULTATOR SEQUENCE FOR STAGE 2
#

process.hltGtStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup           = cms.string("stage2::GTSetup"),
    FedIds          = cms.vint32( 1404 ),
)

process.hltCaloStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup           = cms.string("stage2::CaloSetup"),
    FedIds          = cms.vint32( 1360, 1366 ),
)

process.hltGmtStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage2::GMTSetup"),
    FedIds = cms.vint32(1402),
)

process.hltGtStage2ObjectMap = cms.EDProducer("L1TGlobalProducer",
    MuonInputTag = cms.InputTag("hltGmtStage2Digis","Muon"),
    ExtInputTag = cms.InputTag("hltGtStage2Digis"), # (external conditions are not emulated, use unpacked)
    EtSumInputTag = cms.InputTag("hltCaloStage2Digis", "EtSum"),
    EGammaInputTag = cms.InputTag("hltCaloStage2Digis", "EGamma"),
    TauInputTag = cms.InputTag("hltCaloStage2Digis", "Tau"),
    JetInputTag = cms.InputTag("hltCaloStage2Digis", "Jet"),
    AlgorithmTriggersUnprescaled = cms.bool(True),
    AlgorithmTriggersUnmasked = cms.bool(True),
)

process.HLTL1UnpackerSequence = cms.Sequence(
 process.hltGtStage2Digis +
 process.hltCaloStage2Digis +
 process.hltGmtStage2Digis +
 process.hltGtStage2ObjectMap)

#
# END L1T UNPACKER-EMULATOR SEQUENCE FOR STAGE 2
# ####################################################


# ####################################################
# BEGIN HLT SEED SEQUENCE FOR STAGE 2
#
process.hltL1sL1SingleMuBeamHalo = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1ObjectMapInputTag  = cms.InputTag("hltGtStage2ObjectMap"),
    L1GlobalInputTag     = cms.InputTag("hltGtStage2Digis"),
    L1MuonInputTag       = cms.InputTag("hltGmtStage2Digis","Muon"),
    L1EGammaInputTag     = cms.InputTag("hltCaloStage2Digis","EGamma"),
    L1JetInputTag        = cms.InputTag("hltCaloStage2Digis","Jet"),
    L1TauInputTag        = cms.InputTag("hltCaloStage2Digis","Tau"),
    L1EtSumInputTag      = cms.InputTag("hltCaloStage2Digis","EtSum"),
)

process.hltL1fL1SingleMuBeamHaloFiltered0 = cms.EDFilter( 'HLTMuonL1TFilter',
    PreviousCandTag   =cms.InputTag('hltL1sL1SingleMuBeamHalo'),
    MinPt = cms.double(0.0),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double(2.5),
    CandTag   =cms.InputTag("hltGmtStage2Digis","Muon"),
)

process.HLT_Muon_0  = cms.Sequence( process.hltL1sL1SingleMuBeamHalo + process.hltL1fL1SingleMuBeamHaloFiltered0 )

#process.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    #processName = cms.string( "@" )
#)
process.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
process.HLTTesting  = cms.Sequence(
    process.HLT_Muon_0 +
    process.hltTriggerSummaryRAW
)
#
# END HLT SEED SEQUENCE FOR STAGE 2
# ####################################################

# temp for testing
#process.simGtStage2Digis.PrescaleSet = cms.uint32(4)
#process.simGtStage2Digis.AlgorithmTriggersUnprescaled = cms.bool(False)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
#     fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/g/gflouris/public/SingleMuPt6180_noanti_10k_eta1.root'),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/0A812333-427C-E511-A80A-0025905964A2.root',
        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/1E9D9F9B-467C-E511-85B6-0025905A6090.root',
        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/AA4FBC07-3E7C-E511-B9FC-00261894386C.root',
        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/E2072991-3E7C-E511-803D-002618943947.root',
        '/store/relval/CMSSW_7_6_0_pre7/RelValTTbar_13/GEN-SIM/76X_mcRun2_asymptotic_v9_realBS-v1/00000/FAE20D9D-467C-E511-AF39-0025905B85D8.root'),
    inputCommands = cms.untracked.vstring('keep *',
        'drop *_genParticles_*_*',
        'drop *_genParticlesForJets_*_*',
        'drop *_kt4GenJets_*_*',
        'drop *_kt6GenJets_*_*',
        'drop *_iterativeCone5GenJets_*_*',
        'drop *_ak4GenJets_*_*',
        'drop *_ak7GenJets_*_*',
        'drop *_ak8GenJets_*_*',
        'drop *_ak4GenJetsNoNu_*_*',
        'drop *_ak8GenJetsNoNu_*_*',
        'drop *_genCandidatesForMET_*_*',
        'drop *_genParticlesForMETAllVisible_*_*',
        'drop *_genMetCalo_*_*',
        'drop *_genMetCaloAndNonPrompt_*_*',
        'drop *_genMetTrue_*_*',
        'drop *_genMetIC5GenJs_*_*'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('debug nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-HLTDEBUG'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(1048576),
    fileName = cms.untracked.string('file:step2.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.hlt_step = cms.Path(process.HLTL1UnpackerSequence)
process.hlt_step2 = cms.Path(process.HLTTesting)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# additional tests:
process.dumpED = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpES = cms.EDAnalyzer("PrintEventSetupContent")
process.load('L1Trigger.L1TCommon.l1tSummaryStage2SimDigis_cfi')
process.load('L1Trigger.L1TCommon.l1tSummaryStage2HltDigis_cfi')

process.debug_step = cms.Path(
    process.dumpES +
    process.dumpED +
    process.l1tSummaryStage2SimDigis +
    process.l1tSummaryStage2HltDigis
)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.hlt_step,process.hlt_step2,process.debug_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)
