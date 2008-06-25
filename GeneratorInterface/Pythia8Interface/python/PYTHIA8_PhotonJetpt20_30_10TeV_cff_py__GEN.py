# Auto generated configuration file
# using: 
# $Revision: 1.31 $
# $Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v $
import FWCore.ParameterSet.Config as cms

process = cms.Process('GEN')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/Geometry_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/MagneticField_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.ReleaseValidation = cms.untracked.PSet(
    primaryDatasetName = cms.untracked.string('RelValPYTHIA8_PhotonJetpt20_30_10TeV_cff.pyGEN'),
    totalNumberOfEvents = cms.untracked.int32(5000),
    eventsPerJob = cms.untracked.int32(250)
)
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('PYTHIA8 Photon + Jet for 20 < pT < 30 at 10TeV'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/GenProduction/python/PYTHIA8_PhotonJetpt20_30_10TeV_cff.py,v $')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("Pythia8Source",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.untracked.double(10000.0),
    crossSection = cms.untracked.double(55000000000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring('Main:timesAllowErrors    = 10000', 
            'ParticleDecays:limitTau0 = on', 
            'ParticleDecays:tau0Max   = 10.', 
            'SigmaProcess:alphaSorder = 1', 
            'SigmaProcess:Kfactor     = 1.', 
            'PDF:useLHAPDF            = on', 
            'PDF:LHAPDFset            = cteq6ll.LHpdf', 
            'MultipleInteractions:pT0Ref       = 1.8387', 
            'MultipleInteractions:ecmRef       = 1960.', 
            'MultipleInteractions:ecmPow       = 0.16', 
            'MultipleInteractions:bProfile     = 2', 
            'MultipleInteractions:coreRadius   = 0.4', 
            'MultipleInteractions:coreFraction = 0.5', 
            'BeamRemnants:primordialKT         = on', 
            'BeamRemnants:primordialKThard     = 2.1', 
            'PromptPhoton:all = on', 
            'PhaseSpace:pTHatMin = 20.', 
            'PhaseSpace:pTHatMax = 30.'),
        parameterSets = cms.vstring('processParameters')
    )
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN')
    ),
    fileName = cms.untracked.string('PYTHIA8_PhotonJetpt20_30_10TeV_cff_py__GEN.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    outputCommands = process.RAWSIMEventContent.outputCommands
)

# Other statements
process.GlobalTag.globaltag = 'STARTUP_V1::All'

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.out_step)
