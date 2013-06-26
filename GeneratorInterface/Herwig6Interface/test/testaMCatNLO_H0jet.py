# Auto generated configuration file
# using: 
# Revision: 1.381.2.11 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: Hadronizer_WWTo2L2Nu_mcatnlo_herwig6_8TeV_cff.py -s GEN --conditions=auto:mc --no_exec
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
    input = cms.untracked.int32(200)
)

process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring('/store/lhe/7945/h0j_MT.events_clean.lhe'),
     firstRun = cms.untracked.uint32(1),
     firstLuminosityBlock = cms.untracked.uint32(1),
     skipEvents = cms.untracked.uint32(0),
)


process.options = cms.untracked.PSet(

)

process.generator = cms.EDFilter("Herwig6HadronizerFilter",
        comEnergy = cms.double(8000.0),
        crossSection = cms.untracked.double(-1),
        doMPInteraction = cms.bool(True),
        emulatePythiaStatusCodes = cms.untracked.bool(True),
        filterEfficiency = cms.untracked.double(1.0),
        herwigHepMCVerbosity = cms.untracked.bool(False),
        herwigVerbosity = cms.untracked.int32(1),
        lhapdfSetPath = cms.untracked.string(''),
        maxEventsToPrint = cms.untracked.int32(3),
        printCards = cms.untracked.bool(False),
        useJimmy = cms.bool(True),
        doMatching = cms.untracked.bool(True),
        nMatch = cms.untracked.int32(0),
        inclusiveMatching = cms.untracked.bool(False),
        matchingScale = cms.untracked.double(50.0), 
        ExternalDecays = cms.PSet(
            Photos = cms.untracked.PSet(),
            parameterSets = cms.vstring( "Photos" )
        ),

        HerwigParameters = cms.PSet(
                herwigUEsettings = cms.vstring(
                       'JMUEO     = 2       ! multiparton interaction model',
                       'PTJIM     = 4.189   ! 2.8x(sqrt(s)/1.8TeV)^0.27 @ 8 TeV',
                       'JMRAD(73) = 1.8     ! inverse proton radius squared',
                       'PRSOF     = 0.0     ! prob. of a soft underlying event',
                       'MAXER     = 1000000 ! max error'
                ),
                herwigMcatnlo = cms.vstring(
                        'IPROC      = -1612 ! Higgs to diphoton in this case', 
                        'PTMIN      = 0.5    ! minimum pt in hadronic jet',
                        'MODPDF(1)  = 21100  ! pdf set 1',
                        'MODPDF(2)  = 21100  ! pdf set 2',
                        'RMASS(201) = 125.'
                ),
                parameterSets = cms.vstring('herwigUEsettings',
                                            'herwigMcatnlo')
        )
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('Hadronizer_WWTo2L2Nu_mcatnlo_herwig6_8TeV_cff.py nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('file:output.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

# Path and EndPath definitions
process.generation_step = cms.Path(process.generator*process.pgen)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.endjob_step,process.RECOSIMoutput_step)

