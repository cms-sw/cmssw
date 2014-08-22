# Auto generated configuration file
# using: 
# Revision: 1.381.2.27 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step2 -s VALIDATION -n 10 --filein /store/user/marfin/TTbar_TuneZ2star_13TeV-powheg-tauola/RelValTTbar_8e33_2pt1v4/317184e5e599f32779cb714fb62573b2/output_20_1_zIW.root --conditions auto:mc --mc --customise=HLTriggerOffline/Btag/Validation/customizeHLTBtag.customizeHLTBtag
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
process.load('Configuration.StandardSequences.Validation_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('/store/user/marfin/TTbar_TuneZ2star_13TeV-powheg-tauola/RelValTTbar_8e33_2pt1v4/317184e5e599f32779cb714fb62573b2/output_20_1_zIW.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.27 $'),
    annotation = cms.untracked.string('step2 nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('step2_VALIDATION.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
process.mix.playback = True
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

# Path and EndPath definitions
process.prevalidation_step = cms.Path(process.prevalidation)
process.validation_step = cms.EndPath(process.validation)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.prevalidation_step,process.validation_step,process.endjob_step,process.RECOSIMoutput_step)

# customisation of the process.

# Automatic addition of the customisation function from HLTriggerOffline.Btag.Validation.customizeHLTBtag
from HLTriggerOffline.Btag.Validation.customizeHLTBtag import customizeHLTBtag 

#call to customisation function customizeHLTBtag imported from HLTriggerOffline.Btag.Validation.customizeHLTBtag
process = customizeHLTBtag(process)

# End of customisation functions
