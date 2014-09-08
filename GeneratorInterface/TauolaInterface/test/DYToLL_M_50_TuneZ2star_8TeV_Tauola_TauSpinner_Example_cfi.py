# Auto generated configuration file
# using:
# Revision: 1.381.2.28
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v
# with command line options: DYToLL_M_50_TuneZ2star_8TeV_pythia6_tauola_cff --conditions auto:startup -s GEN,VALIDATION:genvalid_dy --datatier GEN --relval 1000000,20000 -n 1000 --eventcontent RAWSIM
import FWCore.ParameterSet.Config as cms

process = cms.Process('VALIDATION')

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
process.load('Configuration.StandardSequences.Validation_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("GeneratorInterface.TauolaInterface.TauSpinner_cfi")
process.load("GeneratorInterface.TauolaInterface.TauSpinnerFilter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.28 $'),
    annotation = cms.untracked.string('DYToLL_M_50_TuneZ2star_8TeV_pythia6_tauola_cff nevts:1000'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('file:step1.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
generator = cms.PSet(initialSeed = cms.untracked.uint32(12345)),
TauSpinnerGen = cms.PSet(initialSeed = cms.untracked.uint32(12345)),
TauSpinnerZHFilter = cms.PSet(initialSeed = cms.untracked.uint32(429842)),
VtxSmeared = cms.PSet(initialSeed = cms.untracked.uint32(275744))
)


# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
process.mix.playback = True
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            UseTauolaPolarization = cms.bool(False),
            parameterSets = cms.vstring(),
            InputCards = cms.PSet(
                mdtau = cms.int32(0),
                pjak2 = cms.int32(3),
                pjak1 = cms.int32(3)
                )
            ),
        parameterSets = cms.vstring('Tauola')
        ),
                                 maxEventsToPrint = cms.untracked.int32(0),
                                 pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
                                 pythiaHepMCVerbosity = cms.untracked.bool(False),
                                 comEnergy = cms.double(13000.0),
                                 crossSection = cms.untracked.double(762.0),
                                 UseExternalGenerators = cms.untracked.bool(True),
                                 PythiaParameters = cms.PSet(
        pythiaUESettings = cms.vstring('MSTU(21)=1 ! Check on possible errors during program execution',
                                       'MSTJ(22)=2 ! Decay those unstable particles',
                                       'PARJ(71)=10 . ! for which ctau 10 mm',
                                       'MSTP(33)=0 ! no K factors in hard cross sections',
                                       'MSTP(2)=1 ! which order running alphaS',
                                       'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)',
                                       'MSTP(52)=2 ! work with LHAPDF',
                                       'PARP(82)=1.921 ! pt cutoff for multiparton interactions',
                                       'PARP(89)=1800. ! sqrts for which PARP82 is set',
                                       'PARP(90)=0.227 ! Multiple interactions: rescaling power',
                                       'MSTP(95)=6 ! CR (color reconnection parameters)',
                                       'PARP(77)=1.016 ! CR',
                                       'PARP(78)=0.538 ! CR',
                                       'PARP(80)=0.1 ! Prob. colored parton from BBR',
                                       'PARP(83)=0.356 ! Multiple interactions: matter distribution parameter',
                                       'PARP(84)=0.651 ! Multiple interactions: matter distribution parameter',
                                       'PARP(62)=1.025 ! ISR cutoff',
                                       'MSTP(91)=1 ! Gaussian primordial kT',
                                       'PARP(93)=10.0 ! primordial kT-max',
                                       'MSTP(81)=21 ! multiple parton interactions 1 is Pythia default',
                                       'MSTP(82)=4 ! Defines the multi-parton model'),
        processParameters = cms.vstring('MSEL=0 !User defined processes',
                                        'MSUB(1)=1 !Incl Z0/gamma* production',
                                        'MSTP(43)=3 !Both Z0 and gamma*',
                                        'MDME(174,1)=0 !Z decay into d dbar',
                                        'MDME(175,1)=0 !Z decay into u ubar',
                                        'MDME(176,1)=0 !Z decay into s sbar',
                                        'MDME(177,1)=0 !Z decay into c cbar',
                                        'MDME(178,1)=0 !Z decay into b bbar',
                                        'MDME(179,1)=0 !Z decay into t tbar',
                                        'MDME(182,1)=1 !Z decay into e- e+',
                                        'MDME(183,1)=0 !Z decay into nu_e nu_ebar',
                                        'MDME(184,1)=1 !Z decay into mu- mu+',
                                        'MDME(185,1)=0 !Z decay into nu_mu nu_mubar',
                                        'MDME(186,1)=1 !Z decay into tau- tau+',
                                        'MDME(187,1)=0 !Z decay into nu_tau nu_taubar',
                                        'CKIN(1)=50. !Minimum sqrt(s_hat) value (=Z mass)'),
        parameterSets = cms.vstring('pythiaUESettings',
                                    'processParameters')
        )
)


process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen+process.TauSpinnerGen+process.TauSpinnerZHFilter)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.validation_step = cms.EndPath(process.genstepfilter+process.genvalid_dy)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.validation_step,process.endjob_step,process.RAWSIMoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq 
