# Auto generated configuration file
# using: 
# Revision: 1.381.2.13 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: Configuration/GenProduction/python/EightTeV/Hadronizer_MgmMatchTuneZ2star_8TeV_madgraph_tauola_cff.py --filein=file:/afs/cern.ch/work/g/gpetrucc/GENS/MadGraph5_v2_0_0_beta2/ttW_01jets_lo/Events/run_01/unweighted_events.lhe -s GEN,FASTSIM,HLT:GRun --pileup=2012_Summer_inTimeOnly --geometry DB --conditions=auto:startup_GRun --eventcontent=AODSIM --datatier GEN-SIM-DIGI-RECO -n 1000 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('GEN')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(25000))

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.MessageLogger.destinations = ['cout', 'cerr']
process.MessageLogger.cerr.FwkReport.reportEvery = 100

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(1),
    # put here the efficiency of your filter (1. if no filter)
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    # put here the cross section of your process (in pb)
    crossSection = cms.untracked.double(1.0),
    maxEventsToPrint = cms.untracked.int32(1),
    comEnergy = cms.double(8000.0),
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            TauolaPolar,
            TauolaDefaultInputCards
        ),
        parameterSets = cms.vstring('Tauola')
    ),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('PMAS(25,1)=125.0        !mass of Higgs', 
            'MSEL=0                  ! user selection for process', 
            'MSUB(102)=0             !ggH', 
            'MSUB(123)=0             !ZZ fusion to H', 
            'MSUB(124)=0             !WW fusion to H', 
            'MSUB(24)=0              !ZH production', 
            'MSUB(26)=0              !WH production', 
            'MSUB(121)=1             !gg to ttH', 
            'MSUB(122)=1             !qq to ttH', 
            'MDME(210,1)=0           !Higgs decay into dd', 
            'MDME(211,1)=0           !Higgs decay into uu', 
            'MDME(212,1)=0           !Higgs decay into ss', 
            'MDME(213,1)=0           !Higgs decay into cc', 
            'MDME(214,1)=0           !Higgs decay into bb', 
            'MDME(215,1)=0           !Higgs decay into tt', 
            'MDME(216,1)=0           !Higgs decay into', 
            'MDME(217,1)=0           !Higgs decay into Higgs decay', 
            'MDME(218,1)=1           !Higgs decay into e nu e', 
            'MDME(219,1)=1           !Higgs decay into mu nu mu', 
            'MDME(220,1)=1           !Higgs decay into tau nu tau', 
            'MDME(221,1)=0           !Higgs decay into Higgs decay', 
            'MDME(222,1)=1           !Higgs decay into g g', 
            'MDME(223,1)=1           !Higgs decay into gam gam', 
            'MDME(224,1)=1           !Higgs decay into gam Z', 
            'MDME(225,1)=1           !Higgs decay into Z Z', 
            'MDME(226,1)=1           !Higgs decay into W W'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('Configuration/GenProduction/python/EightTeV/Hadronizer_MgmMatchTuneZ2star_8TeV_madgraph_tauola_cff.py nevts:1000'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.AODSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('Pythia_TTH_TuneZ2star_8TeV_tauola.GEN.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RECO')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup_GRun', '')

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.leptonsFromHardScattering1 = cms.EDProducer("GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
        "drop  *  ", # this is the default
        "keep++ (abs(pdgId) = 11 || abs(pdgId) == 13 || abs(pdgId) == 15) && status == 3", 
    )
)
process.leptonsFromHardScattering = cms.EDFilter("GenParticleSelector",
    src = cms.InputTag("leptonsFromHardScattering1"),
    cut = cms.string("status == 1 && (abs(pdgId) == 11 && pt > 7 && abs(eta) < 2.5 ||"+
                                    " abs(pdgId) == 13 && pt > 5 && abs(eta) < 2.4)"),
)
process.dileptonsFromHardScattering = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("leptonsFromHardScattering leptonsFromHardScattering"),
    cut = cms.string("deltaR(daughter(0).eta, daughter(0).phi, daughter(1).eta, daughter(1).phi) > 0.1"),
    checkCharge = cms.bool(False),
)
process.diLepFilterSS = cms.EDFilter("CandViewSelector",
    src = cms.InputTag("dileptonsFromHardScattering"),
    cut = cms.string("charge != 0"), ## if you have tree leptons, at least 2 have the same charge
    filter = cms.bool(True),
)

process.glepSkimSeq = cms.Sequence(
    process.leptonsFromHardScattering1+process.leptonsFromHardScattering+process.dileptonsFromHardScattering+
    process.diLepFilterSS
)

# Path and EndPath definitions
process.generation_step = cms.Path(process.generator + process.pgen + process.glepSkimSeq)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.AODSIMoutput_step = cms.EndPath(process.AODSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step)
process.schedule.extend([process.AODSIMoutput_step])
