import FWCore.ParameterSet.Config as cms

#--------------------------------------

def lumiList( json ):
   import FWCore.PythonUtilities.LumiList as LumiList
   myLumis = LumiList.LumiList(filename = json ).getCMSSWString().split(',')
   return myLumis

def applyJSON( process, json ):

   myLumis = lumiList( json )

   import FWCore.ParameterSet.Types as CfgTypes
   process.source.lumisToProcess = CfgTypes.untracked(CfgTypes.VLuminosityBlockRange())
   process.source.lumisToProcess.extend(myLumis)

#--------------------------------------

process = cms.Process("DataScouting")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.Core.DQM_cfg")

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = 'GR_R_52_V7::All'

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.default.limit = 100
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
## DataScouting PD                         
   fileNames = cms.untracked.vstring(#'/store/data/Run2012B/DataScouting/RAW/v1/000/194/535/16D13DD4-CBA2-E111-AE6F-001D09F24353.root',
                                     #'/store/data/Run2012B/DataScouting/RAW/v1/000/194/983/F6BFCB9C-98A6-E111-A475-003048D2BA82.root',
                                      '/store/data/Run2012B/DataScouting/RAW/v1/000/194/533/FADCCE72-C5A2-E111-825D-003048D2BBF0.root',
                                      '/store/data/Run2012B/DataScouting/RAW/v1/000/194/533/EC8BE038-9DA2-E111-AEEC-00215AEDFD98.root',
                                      '/store/data/Run2012B/DataScouting/RAW/v1/000/194/533/E865F95C-B7A2-E111-9FFF-003048D2BC5C.root',
                                      '/store/data/Run2012B/DataScouting/RAW/v1/000/194/533/E63A7608-99A2-E111-BAD9-001D09F290BF.root'
                                     #'/store/cmst3/user/wreece/CMG/SingleMu/Run2012B-v1/RAW/V00-01-05/outputPhysicsDST_130.root'
                                     #'file:outputPhysicsDST.root'
                                     )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# Un-comment to apply JSON file selection
#applyJSON(process, "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions12/8TeV/Prompt/Cert_190456-204601_8TeV_PromptReco_Collisions12_JSON.txt")

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('DataScouting_DQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQM')
    )
)
#process.DQMoutput.outputCommands.append('keep *_*_*_*')

process.dqmSaver.workflow = '/DataScouting/DQM/Test'
#### UNCOMMENT IF YOU WANT A SINGLE OUTPUT FILE (SHOULD NOT BE USED FOR OFFICIAL DQM WORKFLOWS)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.forceRunNumber = cms.untracked.int32(999999)
####
process.dqmsave_step = cms.Path(process.dqmSaver)

process.load('DQM.DataScouting.dataScouting_cff')

# Path and EndPath definitions
process.testmodule_step = cms.Path(process.dataScoutingDQMSequence)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

process.schedule = cms.Schedule(process.testmodule_step,process.dqmsave_step,process.DQMoutput_step)

