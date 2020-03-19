import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4TrackJets_cfi import ak4TrackJets
from CommonTools.RecoAlgos.TrackWithVertexRefSelector_cfi import *
from RecoJets.JetProducers.TracksForJets_cff import *

recoTrackJetsTask   =cms.Task(trackWithVertexRefSelector,
                              trackRefsForJets,
                              ak4TrackJets )
recoTrackJets   =cms.Sequence(recoTrackJetsTask)

recoAllTrackJets=cms.Task(trackWithVertexRefSelector,
                              trackRefsForJets,
                              ak4TrackJets)
recoAllTrackJets=cms.Sequence(recoAllTrackJets)
