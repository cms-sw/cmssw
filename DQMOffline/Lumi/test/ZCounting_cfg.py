import FWCore.ParameterSet.Config as cms

process = cms.Process('RECODQM')

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
process.GlobalTag.globaltag = '92X_dataRun2_Prompt_v4'

# trigger filter
process.load('HLTrigger/HLTfilters/hltHighLevel_cfi')
process.hltHighLevel.throw = cms.bool(False)
process.hltHighLevel.HLTPaths = cms.vstring()


from CondCore.CondDB.CondDB_cfi import *

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(
'/store/data/Run2017B/SingleMuon/RECO/PromptReco-v1/000/297/218/00000/14C84999-6457-E711-AAE4-02163E0136E0.root'
                                )
                                )
process.source.inputCommands = cms.untracked.vstring("keep *",
                                                         "drop *_MEtoEDMConverter_*_*")

process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(False),
  Rethrow     = cms.untracked.vstring('ProductNotFound'),
  fileMode    = cms.untracked.string('NOMERGE')
  )

process.zcounting = cms.EDAnalyzer('ZCounting',
                                 TriggerEvent    = cms.InputTag('hltTriggerSummaryAOD','','HLT'),
                                 TriggerResults  = cms.InputTag('TriggerResults','','HLT'),
				 edmPVName       = cms.untracked.string('offlinePrimaryVertices'),
                                 edmName       = cms.untracked.string('muons'),
                                 edmTrackName = cms.untracked.string('generalTracks'),

                                 IDType   = cms.untracked.string("Tight"),# Tight, Medium, Loose
                                 IsoType  = cms.untracked.string("NULL"),  # Tracker-based, PF-based
                                 IsoCut   = cms.untracked.double(0.),     # {0.05, 0.10} for Tracker-based, {0.15, 0.25} for PF-based

                                 PtCutL1  = cms.untracked.double(30.0),
                                 PtCutL2  = cms.untracked.double(30.0),
                                 EtaCutL1 = cms.untracked.double(2.4),
                                 EtaCutL2 = cms.untracked.double(2.4),

                                 MassBin  = cms.untracked.int32(50),
                                 MassMin  = cms.untracked.double(66.0),
                                 MassMax  = cms.untracked.double(116.0),

                                 LumiBin  = cms.untracked.int32(2500),
                                 LumiMin  = cms.untracked.double(0.0),
                                 LumiMax  = cms.untracked.double(2500.0),

                                 PVBin    = cms.untracked.int32(60),
                                 PVMin    = cms.untracked.double(0.0),
                                 PVMax    = cms.untracked.double(60.0),

                                 VtxNTracksFitMin = cms.untracked.double(0.),
                                 VtxNdofMin       = cms.untracked.double(4.),
                                 VtxAbsZMax       = cms.untracked.double(24.),
                                 VtxRhoMax        = cms.untracked.double(2.)
                                 )

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
