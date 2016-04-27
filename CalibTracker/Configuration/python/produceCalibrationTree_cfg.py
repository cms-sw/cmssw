import FWCore.ParameterSet.Config as cms

process = cms.Process('CALIB')
process.load('CalibTracker.Configuration.setupCalibrationTree_cff')
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

process.CalibrationTracks.src = 'ALCARECOSiStripCalMinBias'
process.shallowTracks.Tracks  = 'ALCARECOSiStripCalMinBias'

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.add_( cms.Service( "TFileService",
                           fileName = cms.string( 'calibTreeTest.root' ),
                           closeFileFast = cms.untracked.bool(True)  ) )

#process.load('CalibTracker.SiStripCommon.theBigNtuple_cfi')
#process.TkCalPath_SB   = cms.Path( process.theBigNtuple + process.TkCalSeq_StdBunch     )
#process.TkCalPath_SB0T = cms.Path( process.theBigNtuple + process.TkCalSeq_StdBunch0T   )
#process.TkCalPath_IM   = cms.Path( process.theBigNtuple + process.TkCalSeq_IsoMuon      )
#process.TkCalPath_IM0T = cms.Path( process.theBigNtuple + process.TkCalSeq_IsoMuon0T    )
#process.TkCalPath_AB   = cms.Path( process.theBigNtuple + process.TkCalSeq_AagBunch     )
#process.TkCalPath_AB0T = cms.Path( process.theBigNtuple + process.TkCalSeq_AagBunch0T   )
process.TkCalPath_SB   = cms.Path( process.TkCalSeq_StdBunch     )
process.TkCalPath_SB0T = cms.Path( process.TkCalSeq_StdBunch0T   )
process.TkCalPath_IM   = cms.Path( process.TkCalSeq_IsoMuon      )
process.TkCalPath_IM0T = cms.Path( process.TkCalSeq_IsoMuon0T    )
process.TkCalPath_AB   = cms.Path( process.TkCalSeq_AagBunch     )
process.TkCalPath_AB0T = cms.Path( process.TkCalSeq_AagBunch0T   )

#process.TkPathDigi     = cms.Path (process.theBigNtupleDigi)
#process.endPath        = cms.EndPath(process.bigShallowTree)

process.schedule = cms.Schedule( process.TkCalPath_AB, process.TkCalPath_AB0T, 
                                 process.TkCalPath_SB, process.TkCalPath_SB0T,
                                 process.TkCalPath_IM, process.TkCalPath_IM0T )


#following ignored by CRAB
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
process.source = cms.Source (
    "PoolSource",
    fileNames=cms.untracked.vstring(
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60007/869EE593-1FAB-E511-AF99-0025905A60B4.root',
                                    ),
    secondaryFileNames = cms.untracked.vstring())

