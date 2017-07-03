import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('test',eras.Phase2C2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.VtxSmearedNoSmear_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')

# Number of event to generate
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
    )

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('SingleElectronPt50_cfi'),
    name = cms.untracked.string('Applications')
    )

# Output definition
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
        'keep *_hgcalTriggerPrimitiveDigiProducer_*_*'
    ),
    fileName = cms.untracked.string('file:test.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)


# Additional output definition
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("ntuple.root")
    )

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')


# Generator
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *

process.generator = cms.EDFilter(
        "Pythia8PtGun",
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        PGunParameters = cms.PSet(
                ParticleID = cms.vint32(-15),
                AddAntiParticle = cms.bool(True),
                MinPhi = cms.double(-3.14159265359),
                MaxPhi = cms.double(3.14159265359),
                MinPt = cms.double(50.0),
                MaxPt = cms.double(50.0001),
                MinEta = cms.double(1.47),
                MaxEta = cms.double(3.00)
                ),
        pythiaTauJets = cms.vstring(
                'ParticleDecays:sophisticatedTau = 2',
                'ParticleDecays:tauPolarization = 0',
                "15:onMode = off",
                "15:onIfAny = 211 -211 321 -321", # turn on if there is a charged k or pi in the decay products 
                ),
        parameterSets = cms.vstring(
                'pythia8CommonSettings',
                #'pythia8CUEP8M1Settings', # not applicable for taus
                'pythiaTauJets'
                ),
        PythiaParameters = cms.PSet(parameterSets = cms.vstring())
        )

process.mix.digitizers = cms.PSet(process.theDigitizersValid)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)

process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C2d_parameters.seeding_threshold = cms.double(5)
process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].C2d_parameters.clustering_threshold = cms.double(1)

process.hgcl1tpg_step = cms.Path( process.hgcalTriggerPrimitives )
process.digi2raw_step = cms.Path( process.DigiToRaw )

process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Load ntuplizer
process.load('L1Trigger.L1THGCal.hgcalTriggerNtuples_cff')
process.ntuple_step = cms.Path(process.hgcalTriggerNtuples) 

# Draw particle decay tree                                   
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.printTree = cms.EDAnalyzer("ParticleTreeDrawer",
                                   src = cms.InputTag("genParticles"),                                                                 
                                   printP4 = cms.untracked.bool(False),
                                   printPtEtaPhi = cms.untracked.bool(False),
                                   printVertex = cms.untracked.bool(False),
                                   printStatus = cms.untracked.bool(True),
                                   printIndex = cms.untracked.bool(False),
                                   status = cms.untracked.vint32( 1,2,3 )
                                   )
process.drawtree = cms.Path(process.printTree)

# Print particle chain belonging to the hard interaction
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.printDecay = cms.EDAnalyzer("ParticleDecayDrawer",
    src = cms.InputTag("genParticles"),
    printP4 = cms.untracked.bool(False),
    printPtEtaPhi = cms.untracked.bool(False),
    printVertex = cms.untracked.bool(False)
  )

process.drawdecay = cms.Path(process.printDecay)

# Schedule definition
process.schedule = cms.Schedule( process.generation_step, 
                                 # process.drawtree,
                                 #process.drawdecay, 
                                 process.genfiltersummary_step, 
                                 process.simulation_step, 
                                 process.digitisation_step, 
                                 process.L1simulation_step, 
                                 process.hgcl1tpg_step, 
                                 process.digi2raw_step,
                                 process.ntuple_step,
                                 process.endjob_step 
#                                 process.FEVTDEBUGoutput_step
                                 )

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023tilted

#call to customisation function cust_2023tilted imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
###process = cust_2023tilted(process)


