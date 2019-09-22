import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Phase2C2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
#process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
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
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_9_1_1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_91X_upgrade2023_realistic_v1_D17PU200-v1/10000/02ACC76A-C93F-E711-B711-0CC47A4C8EC8.root', 
        '/store/relval/CMSSW_9_1_1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_91X_upgrade2023_realistic_v1_D17PU200-v1/10000/02FD72C1-D53F-E711-BD00-0CC47A7C3420.root', 
        '/store/relval/CMSSW_9_1_1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_91X_upgrade2023_realistic_v1_D17PU200-v1/10000/04EC6BD2-D53F-E711-B2B7-0025905B858E.root', 
        '/store/relval/CMSSW_9_1_1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_91X_upgrade2023_realistic_v1_D17PU200-v1/10000/061B97AF-C93F-E711-A425-0CC47A4C8F12.root', 
        '/store/relval/CMSSW_9_1_1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_91X_upgrade2023_realistic_v1_D17PU200-v1/10000/081DAA75-C93F-E711-8CAF-0025905B85CA.root', 
        '/store/relval/CMSSW_9_1_1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_91X_upgrade2023_realistic_v1_D17PU200-v1/10000/0A1C6B0E-C93F-E711-9145-0CC47A7C3412.root', 
        '/store/relval/CMSSW_9_1_1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_91X_upgrade2023_realistic_v1_D17PU200-v1/10000/0AE954E1-D53F-E711-AA9C-0CC47A4D7632.root', 
        '/store/relval/CMSSW_9_1_1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_91X_upgrade2023_realistic_v1_D17PU200-v1/10000/0C0660A9-CA3F-E711-8583-0CC47A745298.root', 
        '/store/relval/CMSSW_9_1_1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_91X_upgrade2023_realistic_v1_D17PU200-v1/10000/0C66279B-CA3F-E711-A4B3-0CC47A7AB7A0.root', 
        '/store/relval/CMSSW_9_1_1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_91X_upgrade2023_realistic_v1_D17PU200-v1/10000/0E5BF643-D43F-E711-A64C-0CC47A7C34E6.root'),
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
    fileName = cms.untracked.string('file:step3_PU200.root'),
    outputCommands = cms.untracked.vstring( ('keep *') ),
    splitLevel = cms.untracked.int32(0)
)

# debug
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("debugVH_PU200"),
                                    debugModules = cms.untracked.vstring("*"),
                                    categories = cms.untracked.vstring("VectorHitsBuilderValidation"),
                                    debugVH_PU200 = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"),
                                                                       DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                                                       default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                                                       #VectorHitBuilderEDProducer = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
                                                                       #VectorHitBuilderAlgorithm = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
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
    fileName = cms.string('file:VHs_validation_PU200_new.root')
)


# Other statements
process.mix.input.nbPileupEvents.averageNumber = cms.double(200.000000)
process.mix.bunchspace = cms.int32(25)
process.mix.minBunch = cms.int32(-3)
process.mix.maxBunch = cms.int32(3)
process.mix.input.fileNames = cms.untracked.vstring(['/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/0A883B39-083F-E711-8B09-0CC47A7C357A.root', '/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/22E509DA-053F-E711-AA7A-0025905B85BA.root', '/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/3E376DB4-043F-E711-985E-0CC47A74524E.root', '/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/509E21AC-023F-E711-A9F3-0025905B8604.root', '/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/5E53AC15-0A3F-E711-8965-0025905A60E0.root', '/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/62EC67CA-0B3F-E711-81AC-0025905A610C.root', '/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/6476C3E7-063F-E711-B412-0025905B855A.root', '/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/7256251B-0B3F-E711-BBEF-0CC47A78A3B4.root', '/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/8C943DC3-0B3F-E711-BA5F-0CC47A7C34B0.root', '/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/C092AF3B-083F-E711-A0EC-0025905A6070.root', '/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/DE2E5EC5-013F-E711-BE84-0CC47A78A3EC.root', '/store/relval/CMSSW_9_1_1/RelValMinBias_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D17-v1/10000/DE80A4D4-093F-E711-8195-0CC47A4D75F6.root'])
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
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

