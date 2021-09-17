import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process("ALCAISOTRACK",Run2_2017)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HcalIsoTrack=dict()

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_4_0_pre6/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/6EC8FCC8-E2A8-E411-9506-002590596468.root'
 )
)

process.ALCARECOStreamHcalCalIsoTrkFilter = cms.OutputModule("PoolOutputModule",
                                                               SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsoTrkFilter')
        ),
                                                               dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('ALCARECO'),
        filterName = cms.untracked.string('ALCARECOHcalCalIsoTrkFilter')
        ),
                                                             eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
                                                             #                                                            outputCommands = process.OutALCARECOHcalCalIsoTrkFilter.outputCommands,
                                                             outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_gtStage2Digis_*_*',
        'keep *_hbheprereco_*_*',
        'keep *_hbhereco_*_*',
        'keep *_ecalRecHit_*_*',
        'keep *_towerMaker_*_*',
        'keep *_offlineBeamSpot_*_*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep *_TriggerResults_*_*',
        'keep *_generalTracks_*_*',
        'keep *_generalTracksExtra_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_TkAlIsoProdFilter_*_*',
        ),
                                                             fileName = cms.untracked.string('PoolOutput.root'),
                                      )

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamHcalCalIsoTrkFilterOutPath = cms.EndPath(process.ALCARECOStreamHcalCalIsoTrkFilter)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOHcalCalIsoTrkFilter,process.endjob_step,process.ALCARECOStreamHcalCalIsoTrkFilterOutPath)
