import FWCore.ParameterSet.Config as cms 
from Configuration.StandardSequences.Eras import eras



process = cms.Process('DIGI',eras.Phase2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

# Input source
process.source = cms.Source("PoolSource",
       #fileNames = cms.untracked.vstring('/store/relval/CMSSW_9_3_0/RelValSinglePiPt25Eta1p7_2p7/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v2_2023D17noPU-v1/00000/240935CF-1C9B-E711-9F7D-0025905A60BE.root'),
       #fileNames = cms.untracked.vstring('/store/relval/CMSSW_9_3_7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v5_2023D17PU200-v1/10000/F2EAE24E-FD2C-E811-A9B5-0242AC130002.root'),
       fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/j/jsauvan/public/HGCAL/TestingRelVal/CMSSW_9_3_7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/2661406C-972C-E811-9754-0025905A60DE.root'),
       inputCommands=cms.untracked.vstring(
           'keep *',
           'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
           'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
           'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
           'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
           'drop l1tEMTFTrack2016s_simEmtfDigis__HLT'
           )
       )

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('SingleElectronPt10_cfi nevts:10'),
    name = cms.untracked.string('Applications')
)

#---------
# For test Output definition
process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    #outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    outputCommands = cms.untracked.vstring(
        'keep *_*_HGCHitsEE_*',
        'keep *_*_HGCHitsHEback_*',
        'keep *_*_HGCHitsHEfront_*',
        'keep *_mix_*_*',
        'keep *_genParticles_*_*',
	'keep *_hgcalBackEndLayer1Producer_*_*',
	'keep *_hgcalBackEndLayer2Producer_*_*',
	'keep *_hgcalVFEProducer_*_*',
	'keep *_hgcalConcentratorProducer_*_*'
    ),
    fileName = cms.untracked.string('file:test.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW')
    )
    #SelectEvents = cms.untracked.PSet(
    #    SelectEvents = cms.vstring('generation_step')
    #)
)

process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

#---------------------------------


# Output definition
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("ntuple.root")
    )

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# load HGCAL TPG simulation
process.load('L1Trigger.L1THGCal.hgcalVFE_cff')
process.load('L1Trigger.L1THGCal.hgcalConcentrator_cff')
process.load('L1Trigger.L1THGCal.hgcalBackEndLayer1_cff')
process.load('L1Trigger.L1THGCal.hgcalBackEndLayer2_cff')
process.hgcVFE_step = cms.Path(process.hgcalVFE)
process.hgcConcentrator_step = cms.Path(process.hgcalConcentrator)
process.hgcBackEndLayer1_step = cms.Path(process.hgcalBackEndLayer1)
process.hgcBackEndLayer2_step = cms.Path(process.hgcalBackEndLayer2)

# Change to V7 trigger geometry for older samples
#  from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_ZoltanSplit_V7
#  process = custom_geometry_ZoltanSplit_V7(process)

# load ntuplizer
process.load('L1Trigger.L1THGCal.hgcalTriggerNtuples_cff')
process.ntuple_step = cms.Path(process.hgcalTriggerNtuples)

# Schedule definition
#process.schedule = cms.Schedule(process.hgcVFE_step, process.hgcConcentrator_step, process.hgcBackEndLayer1_step, process.hgcBackEndLayer2_step,process.FEVTDEBUGoutput_step)
process.schedule = cms.Schedule(process.hgcVFE_step, process.hgcConcentrator_step, process.hgcBackEndLayer1_step, process.hgcBackEndLayer2_step, process.ntuple_step)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

