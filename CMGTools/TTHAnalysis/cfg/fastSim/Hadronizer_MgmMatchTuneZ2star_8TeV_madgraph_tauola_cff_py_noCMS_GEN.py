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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring('file:/tmp/gpetrucc/unweighted_events.lhe')
    #fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/gpetrucc/w/GENS/MadGraph5_v2_0_0_beta2/ttZ_jets_v2/Events/run_03/unweighted_events.lhe.gz')
    #'/afs/cern.ch/user/g/gpetrucc/w/GENS/MadGraph5_v2_0_0_beta2/ttZ_jets_v2/Events/run_01/unweighted_events.withCMS.lhe')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.MessageLogger.destinations = ['cout', 'cerr']
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

process.generator = cms.EDFilter("Pythia6HadronizerFilter",
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(8000.0),
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            TauolaPolar,
            TauolaDefaultInputCards
        ),
        parameterSets = cms.vstring('Tauola')
    ),
    UseExternalGenerators = cms.untracked.bool(True),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=0         ! User defined processes', 
                        'PMAS(5,1)=4.8   ! b quark mass',
                        'PMAS(6,1)=172.5 ! t quark mass',
			'MSTJ(1)=1       ! Fragmentation/hadronization on or off',
			'MSTP(61)=1      ! Parton showering on or off'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    ),
    jetMatching = cms.untracked.PSet(
       scheme = cms.string("Madgraph"),
       mode = cms.string("auto"),	# soup, or "inclusive" / "exclusive"
       MEMAIN_nqmatch = cms.int32(5),
       MEMAIN_etaclmax = cms.double(5),
       MEMAIN_qcut = cms.double(30),
       MEMAIN_minjets = cms.int32(0),
       MEMAIN_maxjets = cms.int32(99),
       MEMAIN_showerkt = cms.double(0),   
       MEMAIN_excres = cms.string(""),
       outTree_flag = cms.int32(0)    
    )    
)
# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    annotation = cms.untracked.string('Configuration/GenProduction/python/EightTeV/Hadronizer_MgmMatchTuneZ2star_8TeV_madgraph_tauola_cff.py nevts:1000'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.AODSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('/tmp/gpetrucc/ttZ_01jets_LO_TuneZ2star_8TeV_madgraph_tauola.GEN.root'),
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

### Scramble
#import random
#rnd = random.SystemRandom()
#for X in process.RandomNumberGeneratorService.parameterNames_(): 
#   if X != 'saveFileName': getattr(process.RandomNumberGeneratorService,X).initialSeed = rnd.randint(1,99999999)
