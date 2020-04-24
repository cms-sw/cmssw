import FWCore.ParameterSet.Config as cms

from RecoMuon.Configuration.DisplacedMuonSeededStep_cff import *
from RecoMuon.Configuration.preDuplicateMergingDisplacedTracks_cfi import *
from RecoMuon.Configuration.MergeDisplacedTrackCollections_cff import *

iterDisplcedTracking = cms.Sequence(muonSeededStepDisplaced*
                            preDuplicateMergingDisplacedTracks*
                            displacedTracksSequence
                            )
