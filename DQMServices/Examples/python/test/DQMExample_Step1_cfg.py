import os
import FWCore.ParameterSet.Config as cms

release_base = os.environ['CMSSW_RELEASE_BASE']
process = cms.Process('RECODQM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#EvF services
process.FastMonitoringService = cms.Service("FastMonitoringService",
    sleepTime = cms.untracked.int32(1),
    microstateDefPath = cms.untracked.string(
        os.path.join(release_base, 
                     'src/EventFilter/Utilities/plugins/microstatedef.jsd')
        ),
    outputDefPath = cms.untracked.string(
        os.path.join(release_base,
                     'src/EventFilter/Utilities/plugins/output.jsd' )
        ),
    fastName = cms.untracked.string('fastmoni'),
    slowName = cms.untracked.string('slowmoni')
    )

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
    baseDir = cms.untracked.string(os.path.join("/tmp", "data")),
    buBaseDir = cms.untracked.string(os.path.join("/tmp", "data")),
    smBaseDir  = cms.untracked.string(os.path.join("/tmp", "sm")),
    directorIsBu = cms.untracked.bool(False),
    runNumber = cms.untracked.uint32(1)
    )

# my analyzer
process.load('DQMServices.Examples.test.DQMExample_Step1_cfi')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
        #reco from relVals
        'file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/datatest/forTutorial/step2_RAW2DIGI_RECO_fromRelValTTbarLepton.root'
        )
)


process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
                                     fileName = cms.untracked.string("OUT_step1.root"))

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')  #for MC

#DQMStore configuration
process.DQMStore.verbose = cms.untracked.int32(1)

#DQMFileSaver configuration
process.dqmSaver.saveByLumiSection = cms.untracked.int32(1)
process.dqmSaver.convention = cms.untracked.string('FilterUnit')
process.dqmSaver.fileFormat = cms.untracked.string('PB')
process.dqmSaver.workflow = cms.untracked.string('')

# Path and EndPath definitions
process.dqmoffline_step = cms.Path(process.DQMExample_Step1)
process.dqmsave_step = cms.Path(process.DQMSaver)
#process.DQMoutput_step = cms.EndPath(process.DQMoutput)


# Schedule definition
process.schedule = cms.Schedule(
    process.dqmoffline_step,
#    process.DQMoutput_step
    process.dqmsave_step
    )
