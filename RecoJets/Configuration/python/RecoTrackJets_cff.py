import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4TrackJets_cfi import ak4TrackJets
from CommonTools.RecoAlgos.TrackWithVertexRefSelector_cfi import *
from RecoJets.JetProducers.TracksForJets_cff import *


recoTrackJets   =cms.Sequence(trackWithVertexRefSelector+
                              trackRefsForJets+
                              ak4TrackJets
			      )

recoAllTrackJets=cms.Sequence(trackWithVertexRefSelector+
                              trackRefsForJets+
                              ak4TrackJets)
