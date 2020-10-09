import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
#process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
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
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200_rsb-v1/20000/C7A95D39-8FAB-D746-8C06-B11AECA047DD.root',
'/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200_rsb-v1/20000/E4D19FDC-9192-2F44-A4FA-21AB5A93344C.root',
'/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200_rsb-v1/20000/04E557E8-C456-B148-9EC6-2E5E40BD03C6.root',
'/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200_rsb-v1/20000/284ACE29-A506-CD40-9059-899981D7931C.root',
'/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200_rsb-v1/20000/5ACFC231-EBF6-2B4F-A88F-D01171F506CE.root',
'/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200_rsb-v1/20000/8E76FEBA-FF9B-544D-9C17-C5B31777FE39.root',
'/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200_rsb-v1/20000/2958EE82-39A5-154A-9650-5E97AAD44B3E.root',
'/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200_rsb-v1/20000/943CFE41-83ED-F04B-AA74-435821572336.root',
'/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200_rsb-v1/20000/01D965C0-05A4-C543-A676-7C6C67DC2175.root',
'/store/relval/CMSSW_11_2_0_pre3/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200_rsb-v1/20000/7D171FAD-4E5B-824A-A0DF-07175F0A9550.root',
),
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
    outputCommands = process.RECOSIMEventContent.outputCommands,
    #outputCommands = cms.untracked.vstring( ('keep *') ),
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
                                                                       VectorHitsBuilderValidation = cms.untracked.PSet(limit = cms.untracked.int32(-1))
                                                                       )
                                    )

# Analyzer
process.analysis = cms.EDAnalyzer('VectorHitsBuilderValidation',
    src = cms.string("siPhase2Clusters"),
    VH_acc = cms.InputTag("siPhase2VectorHits", "accepted"),
    VH_rej = cms.InputTag("siPhase2VectorHits", "rejected"),
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
process.mix.input.fileNames = cms.untracked.vstring([
'/store/relval/CMSSW_11_2_0_pre5/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v3_2026D49noPU-v1/20000/09AA31DF-737A-B142-A82B-6E48B4B965CD.root',
'/store/relval/CMSSW_11_2_0_pre5/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v3_2026D49noPU-v1/20000/A18897B2-7509-C845-B83A-CEE4AC55FF86.root',
'/store/relval/CMSSW_11_2_0_pre5/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v3_2026D49noPU-v1/20000/184F91C7-EA57-2B49-9E85-D1F494152CDD.root',
'/store/relval/CMSSW_11_2_0_pre5/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v3_2026D49noPU-v1/20000/A505AC6C-6DEA-484C-8FFC-1EA4B54892F5.root',
'/store/relval/CMSSW_11_2_0_pre5/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v3_2026D49noPU-v1/20000/5636AD91-D035-0B44-8D53-2B4FFE6A7722.root',
'/store/relval/CMSSW_11_2_0_pre5/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v3_2026D49noPU-v1/20000/03D1F17E-33EE-A94C-A869-00071F1F45C7.root',
'/store/relval/CMSSW_11_2_0_pre5/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v3_2026D49noPU-v1/20000/EB99E4B3-838A-CC4C-B1B2-8A9D1E1EB453.root',
'/store/relval/CMSSW_11_2_0_pre5/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v3_2026D49noPU-v1/20000/7F44A9B9-38C1-484F-9B4B-DCD0FD7224A8.root',
'/store/relval/CMSSW_11_2_0_pre5/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v3_2026D49noPU-v1/20000/2A3CCF08-CDD1-6246-827F-27CD0CC42478.root',
'/store/relval/CMSSW_11_2_0_pre5/RelValMinBias_14TeV/GEN-SIM-DIGI-RAW/110X_mcRun4_realistic_v3_2026D49noPU-v1/20000/0745B823-5DB0-B74F-B0A6-78E24ED39EE5.root',
])
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

