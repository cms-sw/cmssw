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
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring('/store/lhe/7932/8TeV_ttbarWaMCatNLO_events.lhe'),
#    fileNames = cms.untracked.vstring('file:/tmp/spadhi/8TeV_ttbarWaMCatNLO_events.lhe'),
     firstRun = cms.untracked.uint32(1),
     firstLuminosityBlock = cms.untracked.uint32(1),
     skipEvents = cms.untracked.uint32(0),
)


process.options = cms.untracked.PSet(

)

process.generator = cms.EDFilter("Herwig6HadronizerFilter",
        comEnergy = cms.double(8000.0),
        crossSection = cms.untracked.double(0.1835),
        doMPInteraction = cms.bool(False),
        emulatePythiaStatusCodes = cms.untracked.bool(True),
        filterEfficiency = cms.untracked.double(1.0),
        herwigHepMCVerbosity = cms.untracked.bool(False),
        herwigVerbosity = cms.untracked.int32(0),
        lhapdfSetPath = cms.untracked.string(''),
        maxEventsToPrint = cms.untracked.int32(3),
        printCards = cms.untracked.bool(False),
        useJimmy = cms.bool(False),
        doMatching = cms.untracked.bool(False),
        nMatch = cms.untracked.int32(0),
        inclusiveMatching = cms.untracked.bool(True),
        matchingScale = cms.untracked.double(0.0), 
        ExternalDecays = cms.PSet(
            Photos = cms.untracked.PSet(),
            parameterSets = cms.vstring( "Photos" )
        ),

        HerwigParameters = cms.PSet(
                herwigUEsettings = cms.vstring(
                       'JMUEO     = 1       ! multiparton interaction model',
                       'PTJIM     = 4.189   ! 2.8x(sqrt(s)/1.8TeV)^0.27 @ 8 TeV',
                       'JMRAD(73) = 1.8     ! inverse proton radius squared',
                       'PRSOF     = 0.0     ! prob. of a soft underlying event',
                       'MAXER     = 1000000 ! max error'
                ),
                herwigMcatnlo = cms.vstring(
                        'PTMIN      = 0.5    ! minimum pt in hadronic jet',
                        'IPROC      = -18000  ! proc should be -ve',
                        'MODPDF(1)  = 194800  ! pdf set 1',
                        'MODPDF(2)  = 194800  ! pdf set 2'
                ),
                parameterSets = cms.vstring('herwigUEsettings',
                                            'herwigMcatnlo')
        )
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.4 $'),
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

