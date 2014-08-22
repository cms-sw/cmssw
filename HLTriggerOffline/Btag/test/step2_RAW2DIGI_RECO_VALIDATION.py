# Auto generated configuration file
# using: 
# Revision: 1.381.2.6 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step2 -s RAW2DIGI,RECO,VALIDATION:hltvalidation --eventcontent FEVTDEBUGHLT --conditions auto:startup_GRun --mc --filein file:TTbar_7TeV_cfi_py_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('file:TTbar_7TeV_cfi_py_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    fileName = cms.untracked.string('step2_RAW2DIGI_RECO_VALIDATION.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
process.mix.playback = True
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
process.GlobalTag.globaltag = 'START53_V9::All'
process.GlobalTag.toGet = cms.VPSet()
process.GlobalTag.toGet.append(cms.PSet(tag=cms.string("L1GtTriggerMenu_L1Menu_Collisions2012_v2_mc"),record=cms.string("L1GtTriggerMenuRcd"),connect=cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_L1T"),))
process.GlobalTag.toGet.append(cms.PSet(tag=cms.string("L1GctJetFinderParams_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc"),record=cms.string("L1GctJetFinderParamsRcd"),connect=cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_L1T"),))
process.GlobalTag.toGet.append(cms.PSet(tag=cms.string("L1HfRingEtScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc"),record=cms.string("L1HfRingEtScaleRcd"),connect=cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_L1T"),))
process.GlobalTag.toGet.append(cms.PSet(tag=cms.string("L1HtMissScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc"),record=cms.string("L1HtMissScaleRcd"),connect=cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_L1T"),))
process.GlobalTag.toGet.append(cms.PSet(tag=cms.string("L1JetEtScale_GCTPhysics_2012_04_27_JetSeedThresh5GeV_mc"),record=cms.string("L1JetEtScaleRcd"),connect=cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_L1T"),))
process.GlobalTag.toGet.append(cms.PSet(tag=cms.string("JetCorrectorParametersCollection_AK5PF_2012_V8_hlt_mc"),record=cms.string("JetCorrectionsRecord"),connect=cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),label=cms.untracked.string("AK5PFHLT"),))
process.GlobalTag.toGet.append(cms.PSet(tag=cms.string("JetCorrectorParametersCollection_AK5PFchs_2012_V8_hlt_mc"),record=cms.string("JetCorrectionsRecord"),connect=cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),label=cms.untracked.string("AK5PFchsHLT"),))

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.validation_step = cms.EndPath(process.hltvalidation)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.validation_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)

