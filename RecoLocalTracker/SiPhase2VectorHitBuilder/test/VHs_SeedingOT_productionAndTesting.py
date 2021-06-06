import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#adding only recolocalreco
process.load('RecoLocalTracker.Configuration.RecoLocalTracker_cff')

# import VectorHitBuilder                                                                                                                                                      
process.load('RecoLocalTracker.SiPhase2VectorHitBuilder.siPhase2VectorHits_cfi')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_11_2_0_pre3/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/863B48BB-03ED-0548-AA60-1269291ED1E6.root',
				      '/store/relval/CMSSW_11_2_0_pre3/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/9B6E3A66-B330-8E42-B85E-96A9952A002E.root',
				      '/store/relval/CMSSW_11_2_0_pre3/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/29706FE9-16C9-CE4F-B744-66E07B250D1E.root',
				      '/store/relval/CMSSW_11_2_0_pre3/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/AAB16BEE-B0CE-644A-8E96-35236D793C04.root',
				      '/store/relval/CMSSW_11_2_0_pre3/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/36F89E35-DE2F-174B-95B7-7A9423DED2D8.root',
				      '/store/relval/CMSSW_11_2_0_pre3/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/1BE0A565-4F64-8D42-A5D5-62692F64F0A5.root',
				      '/store/relval/CMSSW_11_2_0_pre3/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/8671B5C9-DE1F-8B4C-8A1C-9C99898FE191.root',
				      '/store/relval/CMSSW_11_2_0_pre3/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/6CA0E490-73F4-1147-906D-050B8B3A3134.root',
				      '/store/relval/CMSSW_11_2_0_pre3/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/AFAB961F-E31F-064D-9031-BEAA10702345.root',
				      '/store/relval/CMSSW_11_2_0_pre3/RelValSingleMuFlatPt2To100/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/2AB6718E-EEA6-494B-AC25-B59CF36DF941.root',
    ),
    secondaryFileNames = cms.untracked.vstring(),
    skipEvents = cms.untracked.uint32(0)
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('file:step3_1event.root'),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Analyzer
process.load('RecoLocalTracker.SiPhase2VectorHitBuilder.vectorHitsBuilderValidation_cfi')

process.TFileService = cms.Service('TFileService',
    fileName = cms.string('file:vh_validation_tilted.root')
)

from RecoTracker.TkSeedGenerator.SeedingOTEDProducer_cfi import SeedingOTEDProducer as _SeedingOTEDProducer
process.phase2SeedingOTEDProducer = _SeedingOTEDProducer.clone()
process.initialStepSeeds = _SeedingOTEDProducer.clone()

process.load('RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEGeometricESProducer_cfi')

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')


# debug
#process.MessageLogger = cms.Service("MessageLogger",
#                                    destinations = cms.untracked.vstring("debugVH_tilted"),
#                                    debugModules = cms.untracked.vstring("*"),
#                                    categories = cms.untracked.vstring("VectorHitBuilderEDProducer","VectorHitBuilderAlgorithm","VectorHitsBuilderValidation","CkfPattern"),
#                                    debugVH_tilted = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"),
#                                                                       DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
#                                                                       default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
#                                                                       VectorHitBuilderEDProducer = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
#                                                                       VectorHitBuilderAlgorithm = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
#                                                                       CkfPattern = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
#                                                                       VectorHitsBuilderValidation = cms.untracked.PSet(limit = cms.untracked.int32(-1))
#                                                                       )
#                                    )


# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.trackerlocalreco_step  = cms.Path(process.trackerlocalreco+process.siPixelClusters+process.siPhase2VectorHits)
process.seedingOT_step  = cms.Path(process.MeasurementTrackerEvent+process.offlineBeamSpot+process.phase2SeedingOTEDProducer)
process.analysis_step = cms.Path(process.vectorHitsBuilderValidation)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.trackerlocalreco_step,process.seedingOT_step,process.RECOSIMoutput_step, process.analysis_step)

# customisation of the process.


# End of customisation functions

