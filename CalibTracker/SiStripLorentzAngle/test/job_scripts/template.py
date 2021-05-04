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
process.GlobalTag.globaltag = 'GLOBAL_TAG::All'

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

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.load("CalibTracker.SiStripLorentzAngle.SiStripLAProfileBooker_cfi")

process.load("Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_cff")
process.ALCARECOTkAlCosmicsCTF0T.src='ALCARECOTkAlCosmicsCTF0T'
process.ALCARECOTkAlCosmicsCTF0T.ptMin=5.

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")
 

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('sistripLAProfile')
process.MessageLogger.cerr.enable = False

process.MessageLogger.files.MY_DEBUG_NUMBER =  cms.untracked.PSet(
      threshold = cms.untracked.string('DEBUG'),
      noLineBreaks = cms.untracked.bool(False),
      DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0))
 )

import FWCore.Python.FileUtils as FileUtils
readFiles = cms.untracked.vstring( FileUtils.loadListFromFile ('JLIST') )

process.source = cms.Source("PoolSource",fileNames = readFiles)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(NEVENT))

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
fileName = cms.untracked.string('file:MY_TMP/MY_HISTOS_HARV_NUMBER.root'),
outputCommands = cms.untracked.vstring('drop *','keep *_MEtoEDMConverter_*_*')
)

process.Schedule = cms.Path(process.seqALCARECOTkAlCosmicsCTF0T*process.LA_analysis)

process.outpath = cms.EndPath(process.myOut)

process.sistripLAProfile.UseStripCablingDB = cms.bool(False)
process.sistripLAProfile.Tracks = 'trackrefitter'
process.sistripLAProfile.TIB_bin = 120
process.sistripLAProfile.TOB_bin = 120
process.sistripLAProfile.SUM_bin = 120
process.sistripLAProfile.fileName = "file:MY_TMP/MY_HISTOS_NUMBER.root"
process.sistripLAProfile.treeName = "file:MY_TMP/MY_TREE_NUMBER.root"




