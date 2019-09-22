import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Phase2C2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#adding only recolocalreco
process.load('RecoLocalTracker.Configuration.RecoLocalTracker_cff')

# import VectorHitBuilder                                                                                                                                                      
process.load('RecoLocalTracker.SiPhase2VectorHitBuilder.SiPhase2VectorHitBuilder_cfi')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_9_1_1/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/91X_upgrade2023_realistic_v1_D17-v1/10000/0A0A27B4-153F-E711-ABC3-0025905A60C6.root', 
        '/store/relval/CMSSW_9_1_1/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/91X_upgrade2023_realistic_v1_D17-v1/10000/34817EB0-163F-E711-83C8-0CC47A7C340C.root', 
        '/store/relval/CMSSW_9_1_1/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/91X_upgrade2023_realistic_v1_D17-v1/10000/3CC9FD4E-173F-E711-B4DF-0025905A60D6.root', 
        '/store/relval/CMSSW_9_1_1/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/91X_upgrade2023_realistic_v1_D17-v1/10000/521169DF-173F-E711-BC3D-0CC47A7C35A4.root', 
        '/store/relval/CMSSW_9_1_1/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/91X_upgrade2023_realistic_v1_D17-v1/10000/6E767157-163F-E711-B315-0025905B8560.root', 
        '/store/relval/CMSSW_9_1_1/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/91X_upgrade2023_realistic_v1_D17-v1/10000/7A9BAA32-173F-E711-B7CA-0CC47A78A496.root', 
        '/store/relval/CMSSW_9_1_1/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/91X_upgrade2023_realistic_v1_D17-v1/10000/BA5F8EE0-173F-E711-97E9-0025905A6122.root', 
        '/store/relval/CMSSW_9_1_1/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/91X_upgrade2023_realistic_v1_D17-v1/10000/CA0AE5B2-153F-E711-B1CE-0025905B85EE.root', 
        '/store/relval/CMSSW_9_1_1/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/91X_upgrade2023_realistic_v1_D17-v1/10000/DEFFF299-173F-E711-9A88-0CC47A4C8EE2.root', 
        '/store/relval/CMSSW_9_1_1/RelValSingleMuPt10Extended/GEN-SIM-DIGI-RAW/91X_upgrade2023_realistic_v1_D17-v1/10000/FA58F15E-163F-E711-B748-0025905A607A.root'),
    #fileNames = cms.untracked.vstring('file:step2.root'),
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
    outputCommands = cms.untracked.vstring( ('keep *') ),
    splitLevel = cms.untracked.int32(0)
)

# debug
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("debugVH_tilted"),
                                    debugModules = cms.untracked.vstring("*"),
                                    categories = cms.untracked.vstring("VectorHitBuilderEDProducer","VectorHitBuilderAlgorithm","VectorHitsBuilderValidation","VectorHitBuilder"),
                                    debugVH_tilted = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"),
                                                                       DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                                                       default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                                                       VectorHitBuilder = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
                                                                       VectorHitBuilderEDProducer = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
                                                                       VectorHitBuilderAlgorithm = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
                                                                       VectorHitsBuilderValidation = cms.untracked.PSet(limit = cms.untracked.int32(-1))
                                                                       )
                                    )

# Analyzer
process.analysis = cms.EDAnalyzer('VectorHitsBuilderValidation',
    src = cms.string("siPhase2Clusters"),
    VH_acc = cms.InputTag("siPhase2VectorHits", "vectorHitsAccepted"),
    VH_rej = cms.InputTag("siPhase2VectorHits", "vectorHitsRejected"),
    CPE = cms.ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"),
    links = cms.InputTag("simSiPixelDigis", "Tracker"),
    trackingParticleSrc = cms.InputTag('mix', 'MergedTrackTruth'),
)
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('file:VHs_validation.root')
)


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.trackerlocalreco_step  = cms.Path(process.trackerlocalreco+process.siPhase2VectorHits)
process.analysis_step = cms.Path(process.analysis)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.trackerlocalreco_step,process.RECOSIMoutput_step, process.analysis_step)

