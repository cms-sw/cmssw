import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from Configuration.ProcessModifiers.convertHGCalDigisSim_cff import convertHGCalDigisSim
from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
process = cms.Process('Produce',Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5/FEVT/PU200_111X_mcRun4_realistic_T15_v1-v1/120000/084C8B72-BC64-DE46-801F-D971D5A34F62.root'
    ),
    inputCommands = cms.untracked.vstring("keep *", 
        "drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT",
        "drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT",
        "drop l1tEMTFHit2016s_simEmtfDigis__HLT",
        "drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT",
        "drop l1tEMTFTrack2016s_simEmtfDigis__HLT",
        'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
        'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
        'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
        'drop l1tEMTFTrack2016s_simEmtfDigis__HLT',
    ),
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# Sequence, Path and EndPath definitions
process.productionSequence = cms.Sequence()

process.load('SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff')
process.load('CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi')

process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.productionSequence += process.SimL1Emulator

############################################################
# Generator-level (visible) hadronic taus
############################################################

process.load("PhysicsTools.JetMCAlgos.TauGenJets_cfi")
process.tauGenJets.GenParticles = cms.InputTag("genParticles")
process.load("PhysicsTools.JetMCAlgos.TauGenJetsDecayModeSelectorAllHadrons_cfi")
process.genTaus = cms.Sequence(process.tauGenJets + process.tauGenJetsSelectorAllHadrons)
process.productionSequence += process.genTaus

############################################################
# produce  L1 HPS PF Tau objects
############################################################

from L1Trigger.Phase2L1Taus.HPSPFTauProducerPF_cfi import HPSPFTauProducerPF
from L1Trigger.Phase2L1Taus.HPSPFTauProducerPuppi_cfi import HPSPFTauProducerPuppi
for useStrips in [ True, False ]:
    for applyPreselection in [ True, False ]:
        moduleNameBase = "HPSPFTauProducer"
        if useStrips and applyPreselection:
            moduleNameBase += "WithStripsAndPreselection"
        elif useStrips and not applyPreselection:
            moduleNameBase += "WithStripsWithoutPreselection"
        elif not useStrips and applyPreselection:
            moduleNameBase += "WithoutStripsWithPreselection"
        elif not useStrips and not applyPreselection:
            moduleNameBase += "WithoutStripsAndPreselection"
        else:
            raise ValueError("Invalid Combination of 'useStrips' and 'applyPreselection' Configuration parameters !!")
        
        moduleNamePF = moduleNameBase + "PF"
        modulePF = HPSPFTauProducerPF.clone(
            useStrips = cms.bool(useStrips),
            applyPreselection = cms.bool(applyPreselection),
            debug = cms.untracked.bool(False)
        )
        setattr(process, moduleNamePF, modulePF)
        process.productionSequence += getattr(process, moduleNamePF)

        moduleNamePuppi = moduleNameBase + "Puppi"
        modulePuppi = HPSPFTauProducerPuppi.clone(
            useStrips = cms.bool(useStrips),
            applyPreselection = cms.bool(applyPreselection),
            debug = cms.untracked.bool(False)
        )
        setattr(process, moduleNamePuppi, modulePuppi)
        process.productionSequence += getattr(process, moduleNamePuppi)


process.production_step = cms.Path(process.productionSequence)

############################################################ 
# write output file
############################################################ 

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("NTuple_HPSPFTauProducer_part_1.root"),                           
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('production_step')
    ),
    outputCommands = cms.untracked.vstring(
        'drop *_*_*_*',                                 
        'keep *_l1tLayer1_PF_*',
        'keep *_l1tLayer1_Puppi_*',
        'keep *_l1tPFProducer*_z0_*',
        'keep *_l1tPFTracksFromL1Tracks*_*_*',
        'keep *_l1tPFClustersFrom*_*_*',
        'keep *_l1tTTTracksFromTracklet_*_*',
        'keep *_l1tVertexProducer_*_*',                                
        'keep *_l1tTkPrimaryVertex_*_*',
        'keep *_slimmedTaus_*_*',
        'keep *_packedPFCandidates_*_*',
        'keep *_generator_*_*',
        'keep *_caloStage2Digis_*_*',
        'keep *_l1tHPSPFTauProducer*PF_*_*',                           
        'keep *_l1tHPSPFTauProducer*Puppi_*_*',                            
        'keep *_prunedGenParticles_*_*',
        'keep *_tauGenJetsSelectorAllHadrons_*_*',
        'keep *_particleFlow_*_*',
        'keep *_generalTracks_*_*',
        'keep *_electronGsfTracks_*_*',
        'keep *_offlineSlimmedPrimaryVertices_*_*',                           
        'keep *_l1tPFTauProducer_*_*',
        'keep *_slimmedAddPileupInfo_*_*', 
        "keep *_l1tPhase1JetProducer_*_*",
    )                           
)
process.outpath = cms.EndPath(process.out)

process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.production_step, process.outpath, process.endjob_step)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

# Enable module run-time report
#process.options = cms.untracked.PSet(
#    wantSummary = cms.untracked.bool(True)
#)

dump_file = open('dump.py','w')
dump_file.write(process.dumpPython())

process.options.numberOfThreads = cms.untracked.uint32(2)
