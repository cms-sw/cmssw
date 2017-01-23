# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: reminiaod --conditions auto:run2_mc --eventcontent MINIAODSIM --filein file:step3.root -s EI:MiniAODfromMiniAOD --datatier MINIAODSIM --runUnscheduled -n 100 --no_exec --fileout file:step4.root --processName MINIAODfromMINIAOD
import FWCore.ParameterSet.Config as cms

process = cms.Process('MINIAODfromMINIAOD')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('PhysicsTools.PatAlgos.slimming.MiniAODfromMiniAOD_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.MessageLogger = cms.Service("MessageLogger",
       destinations   = cms.untracked.vstring('cout'),
       cout           = cms.untracked.PSet(threshold  = cms.untracked.string('ERROR'))
                                    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/RunIISpring16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_miniAODv2_v0-v1/50000/000FF6AC-9F2A-E611-A063-0CC47A4C8EB0.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('reminiaod nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.MINIAODSIMoutput = cms.OutputModule("PoolOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(4),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('MINIAODSIM'),
        filterName = cms.untracked.string('')
    ),
    dropMetaData = cms.untracked.string('ALL'),
    eventAutoFlushCompressedSize = cms.untracked.int32(15728640),
    fastCloning = cms.untracked.bool(False),
    fileName = cms.untracked.string('file:step4.root'),
    outputCommands = process.MINIAODSIMEventContent.outputCommands,
    overrideInputFileSplitLevels = cms.untracked.bool(True)
)

# Additional output definition

# The regressions are in the conditions database starting at this versions
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Other access methods:

# Apply the regression from local sqlite file
#from EgammaAnalysis.ElectronTools.regressionWeights_local_cfi import GBRDWrapperRcd
#process.regressions           = GBRDWrapperRcd
#process.es_prefer_regressions = cms.ESPrefer('PoolDBESSource','regressions')
#process.load('EgammaAnalysis.ElectronTools.regressionApplication_cff')
#process.EGMenergyCorrection = cms.Path(process.regressionApplication)

# Apply the regression from a remote database
from EgammaAnalysis.ElectronTools.regressionWeights_cfi import regressionWeights
process = regressionWeights(process)

process.load('EgammaAnalysis.ElectronTools.regressionApplication_cff')
process.EGMenergyCorrection = cms.Path(process.regressionApplication)

# Path and EndPath definitions

# You don't need to really remak all of miniAOD, only the regressions
#process.eventinterpretaion_step = cms.Path(process.EIsequence)

process.endjob_step = cms.EndPath(process.endOfProcess)
process.MINIAODSIMoutput_step = cms.EndPath(process.MINIAODSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.EGMenergyCorrection,process.endjob_step,process.MINIAODSIMoutput_step)

