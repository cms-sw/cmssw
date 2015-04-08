# Auto generated configuration file
# using: 
# Revision: 1.381.2.28 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step4 --conditions INSERT_GT_HERE -s RAW2DIGI,L1Reco,RECO --scenario HeavyIons --datatier GEN-SIM-RECO --himix --eventcontent RECODEBUG -n 100 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')
process.load('SimGeneral.MixingModule.HiEventMixing_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
                            secondaryFileNames = cms.untracked.vstring(),
                            fileNames = cms.untracked.vstring("/store/user/pkurt/Hydjet1p8_TuneDrum_Quenched_MinBias_2760GeV/HydjetDrum_Pyquen_Dijet80_Embedded_DIGI_d20140128_STARTHI53_LV1_Track7_Jet1/e69cac9dd0f8008cc653446abec02c5d/step3_DIGI_L1_DIGI2RAW_HLT_RAW2DIGI_L1Reco_9_1_ii4.root",
                                                              #    "root://xrootd.cmsaf.mit.edu//store/user/pkurt/Hydjet1p8_TuneDrum_Quenched_MinBias_2760GeV/HydjetDrum_Pyquen_Dijet80_Embedded_DIGI_d20140128_STARTHI53_LV1_Track7_Jet1/e69cac9dd0f8008cc653446abec02c5d/step3_DIGI_L1_DIGI2RAW_HLT_RAW2DIGI_L1Reco_9_1_ii4.root"
                                                              )
                            )

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.28 $'),
    annotation = cms.untracked.string('step4 nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.RECODEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RECODEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('MC_RECO.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RECO')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'STARTHI53_LV1::All', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstructionHeavyIons)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECODEBUGoutput_step = cms.EndPath(process.RECODEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step,process.RECODEBUGoutput_step)



process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                      oncePerEventMode=cms.untracked.bool(False))

process.Timing=cms.Service("Timing",
                           useJobReport = cms.untracked.bool(True)
                           )

# Load Beamspot
from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
process.beamspot = cms.ESSource("PoolDBESSource",CondDBSetup,
                                toGet = cms.VPSet(cms.PSet( record = cms.string('BeamSpotObjectsRcd'),
                                                            tag= cms.string('Realistic2p76TeVCollisions2013_START53_V9_v1_mc')
                                                            )),
                                connect =cms.string('frontier://FrontierProd/CMS_COND_31X_BEAMSPOT')
                                )
process.es_prefer_beamspot = cms.ESPrefer("PoolDBESSource","beamspot")
