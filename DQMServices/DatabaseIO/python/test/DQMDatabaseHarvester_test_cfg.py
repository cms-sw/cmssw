import FWCore.ParameterSet.Config as cms

process = cms.Process('DQMANDHARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')  #for MC

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.DQMStore.verbose = cms.untracked.int32(4)

process.maxEvents = cms.untracked.PSet(
	input = cms.untracked.int32(200)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
        #reco from relVals
        'file:/build/aandrzej/CMSSW_7_6_0_pre1/src/DQMServices/Examples/python/test/1324.0_TTbarLepton_13+TTbarLepton_13INPUT+DIGIUP15+RECOUP15+HARVESTUP15+MINIAODMCUP15/step3.root'
        )
)

# my analyzer
process.load('DQMServices.Examples.test.DQMExample_Step1_cfi')

# my client and my Tests
process.load('DQMServices.Examples.test.DQMExample_Step2DB_cfi')
process.load('DQMServices.Examples.test.DQMExample_GenericClient_cfi')
process.load('DQMServices.Examples.test.DQMExample_qTester_cfi')
process.dqmmodules = cms.Sequence(process.dqmSaver)

# Path and EndPath definitions
process.dqmoffline_step = cms.Path(process.DQMExample_Step1)
process.myHarvesting = cms.Path(process.DQMExample_Step2DB)
process.myEff = cms.Path(process.DQMExample_GenericClient)
process.myTest = cms.Path(process.DQMExample_qTester)
process.dqmsave_step = cms.Path(process.dqmmodules)

# Schedule definition
process.schedule = cms.Schedule(
								process.dqmoffline_step,
								process.myEff,
								process.myTest,
								process.myHarvesting,
								process.dqmsave_step
    )
process.dqmSaver.workflow = '/TTbarLepton/myTest/DQM'

