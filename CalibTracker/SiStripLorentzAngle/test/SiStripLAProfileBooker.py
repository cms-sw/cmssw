import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

import CalibTracker.Configuration.Common.PoolDBESSource_cfi

process = cms.Process("analyze")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Configuration.StandardSequences.GeometryIdeal_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'CRAFT_31X::All'

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.Timing = cms.Service("Timing")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
    
#process.SiStripLorentzAngle = cms.ESSource("PoolDBESSource",
#    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#    DBParameters = cms.PSet(messageLevel = cms.untracked.int32(2),
#                            authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#                            ),
#    toGet = cms.VPSet(cms.PSet(record = cms.string('SiStripLorentzAngleRcd'),
#                               tag = cms.string('SiStripLA_TEST_Layers')
#    )),
#    connect = cms.string('sqlite_file:DB_LA_TEST_Layers.db')
#)
#                                      
#process.es_prefer_SiStripLorentzAngle   = cms.ESPrefer("PoolDBESSource","SiStripLorentzAngle")

process.load("RecoTracker.TrackProducer.TrackRefitter_cff")

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.load("CalibTracker.SiStripLorentzAngle.SiStripLAProfileBooker_cfi")

process.load("Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_cff")
process.ALCARECOTkAlCosmicsCTF0T.src='ALCARECOTkAlCosmicsCTF0T'
process.ALCARECOTkAlCosmicsCTF0T.ptMin=5.

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")
 
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('sistripLAProfile')
process.MessageLogger.cerr.enable = False

process.MessageLogger.files.LAProfileDebug =  cms.untracked.PSet(
      threshold = cms.untracked.string('DEBUG'),
      noLineBreaks = cms.untracked.bool(False),
      DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0))
 )

import FWCore.Python.FileUtils as FileUtils

#readFiles = cms.untracked.vstring( FileUtils.loadListFromFile ('source_list_test.txt') )
#process.source = cms.Source("PoolSource",fileNames = readFiles)

process.source = cms.Source ("PoolSource",fileNames=cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_ALL_V4_StreamALCARECOTkAlCosmics0T_step3_AlcaReco-v2/0052/529AA4E7-08CC-DD11-A489-001D0967D21A.root'),)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.trackrefitter = cms.EDFilter("TrackRefitter",
    src = cms.InputTag("ALCARECOTkAlCosmicsCTF0T"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    constraint = cms.string(''),
    srcConstr  = cms.InputTag(''),
    Fitter = cms.string('RKFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('ctf'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

process.LA_analysis = cms.Sequence(process.offlineBeamSpot*process.trackrefitter*process.sistripLAProfile*process.MEtoEDMConverter)

process.myOut = cms.OutputModule("PoolOutputModule",
fileName = cms.untracked.string('file:LA_Histos_Harv_NEWAL.root'),
outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_*')
)

process.Schedule = cms.Path(process.seqALCARECOTkAlCosmicsCTF0T*process.LA_analysis)

process.outpath = cms.EndPath(process.myOut)

process.sistripLAProfile.UseStripCablingDB = cms.bool(False)
process.sistripLAProfile.Tracks = 'trackrefitter'
process.sistripLAProfile.TIB_bin = 120
process.sistripLAProfile.TOB_bin = 120
process.sistripLAProfile.SUM_bin = 120
process.sistripLAProfile.fileName = "file:LA_Histos_NEWAL.root"
process.sistripLAProfile.treeName = "file:LA_Trees_NEWAL.root"




