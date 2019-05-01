import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
process = cms.Process('RECODQM', Run2_2018)

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryDB_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_condDBv2_cff')
process.load('Configuration/EventContent/EventContent_cff')
process.load('TrackingTools/TransientTrack/TransientTrackBuilder_cfi')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.GlobalTag.globaltag = '101X_dataRun2_Prompt_v9'
#process.GlobalTag.globaltag = '92X_dataRun2_Prompt_v4'


# trigger filter
process.load('HLTrigger/HLTfilters/hltHighLevel_cfi')
process.hltHighLevel.throw = cms.bool(False)
process.hltHighLevel.HLTPaths = cms.vstring()


from CondCore.CondDB.CondDB_cfi import *

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(
#'/store/data/Run2017B/SingleMuon/RECO/PromptReco-v1/000/297/218/00000/14C84999-6457-E711-AAE4-02163E0136E0.root'
'file:/eos/cms/store/data/Run2018D/SingleMuon/AOD/22Jan2019-v2/110000/104D2C01-12D2-0742-BE61-897755323EDE.root'
                                )
                                )

process.source.inputCommands = cms.untracked.vstring("keep *",
                                                         "drop *_MEtoEDMConverter_*_*")

process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(False),
  Rethrow     = cms.untracked.vstring('ProductNotFound'),
  fileMode    = cms.untracked.string('FULLMERGE')
  )


from DQMOffline.Lumi.ZCounting_cff import zcounting

process.load("DQMOffline.Lumi.ZCounting_cff") 

#process.zcounting.MuonTriggerNames = cms.vstring("HLT_IsoMu27_v*","HLT_IsoMu24_v*")
#process.zcounting.MuonTriggerObjectNames = cms.vstring("hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07",
#                                                      "hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07")

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
                                     fileName = cms.untracked.string("OUT_step1.root"))

# Path and EndPath definitions
process.dqmoffline_step = cms.Path(process.zcounting)
process.dqmsave_step = cms.Path(process.DQMSaver)
#process.DQMoutput_step = cms.EndPath(process.DQMoutput)


# Schedule definition
process.schedule = cms.Schedule(
    process.dqmoffline_step,
#    process.DQMoutput_step
    process.dqmsave_step
    )

process.dqmSaver.workflow = '/SingleMuon/Run2017B-PromptReco-v1/RECO'
