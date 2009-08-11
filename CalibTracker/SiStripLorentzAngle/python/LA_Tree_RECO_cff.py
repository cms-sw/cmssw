import FWCore.ParameterSet.Config as cms

# import of standard configurations
from Configuration.StandardSequences.Services_cff import *
from FWCore.MessageService.MessageLogger_cfi import *
from Configuration.StandardSequences.MixingNoPileUp_cff import *
from Configuration.StandardSequences.GeometryIdeal_cff import *
from Configuration.StandardSequences.MagneticField_38T_cff import *
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *

#tracking
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
from RecoTracker.Configuration.RecoTracker_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoPixelVertexing.Configuration.RecoPixelVertexing_cff import *
recotrack = cms.Sequence( offlineBeamSpot + siPixelRecHits*siStripMatchedRecHits*recopixelvertexing*ckftracks)

#ntuple
from UserCode.ShallowTools.ShallowClustersProducer_cfi import *
from UserCode.ShallowTools.ShallowTrackClustersProducer_cfi import *
from UserCode.ShallowTools.ShallowTracksProducer_cfi import *
calibrationTree = cms.EDAnalyzer("ShallowTree",
                             outputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_*_clusterdetid_*',
    'keep *_*_clusterwidth_*',
    'keep *_*_clustervariance_*',
    'keep *_*_tsostrackmulti_*',
    'keep *_*_tsostrackindex_*',
    'keep *_*_tsosdriftx_*',
    'keep *_*_tsosdriftz_*',
    'keep *_*_tsoslocalpitch_*',
    'keep *_*_tsoslocaltheta_*',
    'keep *_*_tsoslocalphi_*',
    'keep *_*_tsosBdotYhat_*',
    'keep *_*_trackchi2ndof_*',
    'keep *_*_trackhitsvalid_*'
    ))
theBigNtuple = cms.Sequence( (shallowClusters +
                              shallowTracks +
                              shallowTrackClusters) *
                             calibrationTree
                             )

#Schedule
recotrack_step  = cms.Path( recotrack )
ntuple_step     = cms.Path( theBigNtuple )
schedule = cms.Schedule( recotrack_step, ntuple_step )
