# Auto generated configuration file
# using: 
# Revision: 1.13 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: WHToTauTau_M125GeV_14TeV_cff.py -s GEN,SIM,DIGI,L1,DIGI2RAW --conditions POSTLS161_V12::All --geometry Geometry/GEMGeometry/cmsExtendedGeometryPostLS1plusGEMXML_cfi --datatier GEN-SIM-RAW --eventcontent RAWSIM -n 10 --no_exec --fileout out_sim.root --pileup NoPileUp --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.digiCustoms
import FWCore.ParameterSet.Config as cms

process = cms.Process('DIGI2RAW')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.13 $'),
    annotation = cms.untracked.string('WHToTauTau_M125GeV_14TeV_cff.py nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('out_sim.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RAW')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'POSTLS161_V12::All', '')

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            UseTauolaPolarization = cms.bool(True),
            InputCards = cms.PSet(
                mdtau = cms.int32(0),
                pjak2 = cms.int32(0),
                pjak1 = cms.int32(0)
            )
        ),
        parameterSets = cms.vstring('Tauola')
    ),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(14000.0),
    crossSection = cms.untracked.double(0.0950528),
    UseExternalGenerators = cms.untracked.bool(True),
    PythiaParameters = cms.PSet(
        pythiaUESettings = cms.vstring('MSTU(21)=1     ! Check on possible errors during program execution', 
            'MSTJ(22)=2     ! Decay those unstable particles', 
            'PARJ(71)=10 .  ! for which ctau  10 mm', 
            'MSTP(33)=0     ! no K factors in hard cross sections', 
            'MSTP(2)=1      ! which order running alphaS', 
            'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)', 
            'MSTP(52)=2     ! work with LHAPDF', 
            'PARP(82)=1.832 ! pt cutoff for multiparton interactions', 
            'PARP(89)=1800. ! sqrts for which PARP82 is set', 
            'PARP(90)=0.275 ! Multiple interactions: rescaling power', 
            'MSTP(95)=6     ! CR (color reconnection parameters)', 
            'PARP(77)=1.016 ! CR', 
            'PARP(78)=0.538 ! CR', 
            'PARP(80)=0.1   ! Prob. colored parton from BBR', 
            'PARP(83)=0.356 ! Multiple interactions: matter distribution parameter', 
            'PARP(84)=0.651 ! Multiple interactions: matter distribution parameter', 
            'PARP(62)=1.025 ! ISR cutoff', 
            'MSTP(91)=1     ! Gaussian primordial kT', 
            'PARP(93)=10.0  ! primordial kT-max', 
            'MSTP(81)=21    ! multiple parton interactions 1 is Pythia default', 
            'MSTP(82)=4     ! Defines the multi-parton model'),
        processParameters = cms.vstring('MSEL=0            ! User defined processes', 
            'MSUB(26)= 1       ! gg->WH (SM)', 
            'PMAS(25,1)= 125.  ! m_h', 
            'PMAS(6,1)= 172.6  ! mass of top quark', 
            'PMAS(23,1)=91.187 ! mass of Z', 
            'PMAS(24,1)=80.39  ! mass of W', 
            'MDME(190,1)=0   !W decay into dbar u', 
            'MDME(191,1)=0   !W decay into dbar c', 
            'MDME(192,1)=0   !W decay into dbar t', 
            'MDME(194,1)=0   !W decay into sbar u', 
            'MDME(195,1)=0   !W decay into sbar c', 
            'MDME(196,1)=0   !W decay into sbar t', 
            'MDME(198,1)=0   !W decay into bbar u', 
            'MDME(199,1)=0   !W decay into bbar c', 
            'MDME(200,1)=0   !W decay into bbar t', 
            'MDME(205,1)=0   !W decay into bbar tp', 
            'MDME(206,1)=1   !W decay into e+ nu_e', 
            'MDME(207,1)=1   !W decay into mu+ nu_mu', 
            'MDME(208,1)=1   !W decay into tau+ nu_tau', 
            'MDME(210,1)=0   !Higgs decay into dd', 
            'MDME(211,1)=0   !Higgs decay into uu', 
            'MDME(212,1)=0   !Higgs decay into ss', 
            'MDME(213,1)=0   !Higgs decay into cc', 
            'MDME(214,1)=0   !Higgs decay into bb', 
            'MDME(215,1)=0   !Higgs decay into tt', 
            'MDME(216,1)=0   !Higgs decay into', 
            'MDME(217,1)=0   !Higgs decay into Higgs decay', 
            'MDME(218,1)=0   !Higgs decay into e nu e', 
            'MDME(219,1)=0   !Higgs decay into mu nu mu', 
            'MDME(220,1)=1   !Higgs decay into tau tau', 
            'MDME(221,1)=0   !Higgs decay into Higgs decay', 
            'MDME(222,1)=0   !Higgs decay into g g', 
            'MDME(223,1)=0   !Higgs decay into gam gam', 
            'MDME(224,1)=0   !Higgs decay into gam Z', 
            'MDME(225,1)=0   !Higgs decay into Z Z', 
            'MDME(226,1)=0   !Higgs decay into W W'),
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)


process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.endjob_step,process.RAWSIMoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq 

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import digiCustoms 

#call to customisation function digiCustoms imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = digiCustoms(process)

# End of customisation functions
