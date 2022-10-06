import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTVAL")


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#process.MessageLogger.categories.append("HLTTauDQMOffline")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#'/store/data/Run2018C/SingleMuon/RAW-RECO/ZMu-PromptReco-v2/000/319/756/00000/FEF9093D-3C8B-E811-8048-FA163EC8DFDC.root'
#	'file:step3.root'
#        'file:/afs/cern.ch/work/s/slehti/Validation/TauTriggerValidation/step2_RAW2DIGI_L1Reco_RECO_0.root'
        'file:/eos/user/b/ballmond/DQM/forSami_RECO/step2_RAW2DIGI_L1Reco_RECO_0.root'
    )
)

# Set GlobalTag (automatically)
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

#Load DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")


#Reconfigure Environment and saver
#process.dqmEnv.subSystemFolder = cms.untracked.string('HLT/HLTTAU')
#process.DQM.collectorPort = 9091
#process.DQM.collectorHost = cms.untracked.string('pcwiscms10')

process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.workflow = cms.untracked.string('/A/N/C')
process.dqmSaver.forceRunNumber = cms.untracked.int32(123)


#Load the Validation
process.load("DQMOffline.Trigger.HLTTauDQMOffline_cff")
process.hltTauOfflineMonitor_PFTaus.PlotLevel = cms.untracked.int32(1)
process.hltTauOfflineMonitor_Inclusive.PlotLevel = cms.untracked.int32(1)
process.hltTauOfflineMonitor_TagAndProbe.PlotLevel = cms.untracked.int32(1)

#Load The Post processor
process.load("DQMOffline.Trigger.HLTTauPostProcessor_cfi")
#process.load("DQMOffline.Trigger.HLTTauQualityTester_cfi")


#Define the Paths
process.validation = cms.Path(process.HLTTauDQMOffline)
#process.validation = cms.Path(process.HLTTauVal)

#process.postProcess = cms.EndPath(process.HLTTauPostVal+process.hltTauRelvalQualityTests+process.dqmSaver)
process.postProcess = cms.EndPath(process.HLTTauPostSeq+process.dqmSaver)
#process.postProcess = cms.EndPath(process.dqmSaver)
process.schedule =cms.Schedule(process.validation,process.postProcess)



process.output = cms.OutputModule("PoolOutputModule",
   outputCommands = cms.untracked.vstring(
       "keep *",
   ),
   fileName = cms.untracked.string("CMSSW.root")
)     
#process.out_step = cms.EndPath(process.output)
#process.schedule =cms.Schedule(process.validation,process.postProcess,process.out_step)

