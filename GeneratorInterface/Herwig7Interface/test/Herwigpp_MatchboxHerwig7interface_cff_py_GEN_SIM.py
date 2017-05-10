# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Herwigpp_DummyProcess_cff.py --fileout file:DummyProcess.root --mc --eventcontent RAWSIM --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1,Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM --conditions MCRUN2_71_V1::All --beamspot Realistic50ns13TeVCollision --step GEN,SIM --magField 38T_PostLS1 --no_exec -n 5
import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')


process.MessageLogger = cms.Service("MessageLogger",
       destinations   = cms.untracked.vstring(
                                             'detailedInfo'
                                               ,'critical'
                                               ,'cerr'
                    ),
       critical       = cms.untracked.PSet(
                        threshold = cms.untracked.string('ERROR') 
        ),
       detailedInfo   = cms.untracked.PSet(
                      threshold  = cms.untracked.string('INFO') 
       ),
       cerr           = cms.untracked.PSet(
                       threshold  = cms.untracked.string('WARNING') 
        )
)

process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('Herwigpp_DummyProcess_cff.py nevts:5'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('file:TestProcess.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'MCRUN2_71_V1::All', '')

process.generator = cms.EDFilter("Herwig7GeneratorFilter",
    hwpp_cmsDefaults = cms.vstring('+hwpp_basicSetup', 
        '+hwpp_setParticlesStableForDetector'),
    run = cms.string('InterfaceMatchboxTest'),
   # dumpConfig = cms.untracked.string('HerwigConfig.in'),
    repository = cms.string('HerwigDefaults.rpo'),
    dataLocation = cms.string('${HERWIGPATH}'),
    hwpp_setParticlesStableForDetector = cms.vstring('set /Herwig/Particles/mu-:Stable Stable', 
        'set /Herwig/Particles/mu+:Stable Stable', 
        'set /Herwig/Particles/Sigma-:Stable Stable', 
        'set /Herwig/Particles/Sigmabar+:Stable Stable', 
        'set /Herwig/Particles/Lambda0:Stable Stable', 
        'set /Herwig/Particles/Lambdabar0:Stable Stable', 
        'set /Herwig/Particles/Sigma+:Stable Stable', 
        'set /Herwig/Particles/Sigmabar-:Stable Stable', 
        'set /Herwig/Particles/Xi-:Stable Stable', 
        'set /Herwig/Particles/Xibar+:Stable Stable', 
        'set /Herwig/Particles/Xi0:Stable Stable', 
        'set /Herwig/Particles/Xibar0:Stable Stable', 
        'set /Herwig/Particles/Omega-:Stable Stable', 
        'set /Herwig/Particles/Omegabar+:Stable Stable', 
        'set /Herwig/Particles/pi+:Stable Stable', 
        'set /Herwig/Particles/pi-:Stable Stable', 
        'set /Herwig/Particles/K+:Stable Stable', 
        'set /Herwig/Particles/K-:Stable Stable', 
        'set /Herwig/Particles/K_S0:Stable Stable', 
        'set /Herwig/Particles/K_L0:Stable Stable'),
    generatorModule = cms.string('/Herwig/Generators/EventGenerator'),
    eventHandlers = cms.string('/Herwig/EventHandlers'),
    hwpp_basicSetup = cms.vstring('#read Matchbox/GenericCollider.in',
	'#create ThePEG::RandomEngineGlue /Herwig/RandomGlue', 
        '#set /Herwig/Generators/EventGenerator:RandomNumberGenerator /Herwig/RandomGlue', 
        'set /Herwig/Generators/EventGenerator:DebugLevel 2', 
        'set /Herwig/Generators/EventGenerator:PrintEvent 1', 
        'set /Herwig/Generators/EventGenerator:MaxErrors 10000'),
    configFiles = cms.vstring(),
    crossSection = cms.untracked.double(-1),
    parameterSets = cms.vstring(
        'Matchbox',
	'hwpp_cmsDefaults'),
    filterEfficiency = cms.untracked.double(1.0),
    Matchbox = cms.vstring('read Matchbox/PPCollider.in',
	'cd /Herwig/EventHandlers',
	'set EventHandler:LuminosityFunction:Energy 13000*GeV',
	'## Model assumptions',
	'read Matchbox/StandardModelLike.in',
	'read Matchbox/DiagonalCKM.in',
	'## Set the order of the couplings',
	'cd /Herwig/MatrixElements/Matchbox',
	'set Factory:OrderInAlphaS 0',
	'set Factory:OrderInAlphaEW 2',
	'## Select the process',
	'do Factory:Process p p -> e+ e-',
	'# read Matchbox/MadGraph-GoSam.in',
	'# read Matchbox/MadGraph-MadGraph.in',
	'# read Matchbox/MadGraph-OpenLoops.in',
	'set /Herwig/Cuts/ChargedLeptonPairMassCut:MinMass 60*GeV',
	'set /Herwig/Cuts/ChargedLeptonPairMassCut:MaxMass 120*GeV',
	'cd /Herwig/MatrixElements/Matchbox',
	'set Factory:ScaleChoice /Herwig/MatrixElements/Matchbox/Scales/LeptonPairMassScale',
	'read Matchbox/MCatNLO-DefaultShower.in',
	'# read Matchbox/NLO-NoShower.in',
	'# read Matchbox/LO-NoShower.in',
	'read Matchbox/FiveFlavourScheme.in',
	'read Matchbox/MMHT2014.in',
	'do /Herwig/MatrixElements/Matchbox/Factory:ProductionMode',
    )
)


process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.RAWSIMoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq 

# customisation of the process.

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring 

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

# End of customisation functions
