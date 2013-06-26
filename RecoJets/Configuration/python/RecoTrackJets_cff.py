import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.sc5TrackJets_cfi import sisCone5TrackJets
from RecoJets.JetProducers.ic5TrackJets_cfi import iterativeCone5TrackJets
from RecoJets.JetProducers.ak5TrackJets_cfi import ak5TrackJets
from RecoJets.JetProducers.gk5TrackJets_cfi import gk5TrackJets
from RecoJets.JetProducers.kt4TrackJets_cfi import kt4TrackJets
from RecoJets.JetProducers.ca4TrackJets_cfi import ca4TrackJets
from CommonTools.RecoAlgos.TrackWithVertexRefSelector_cfi import *
from RecoJets.JetProducers.TracksForJets_cff import *


sisCone7TrackJets = sisCone5TrackJets.clone( rParam = 0.7 )
ak7TrackJets = ak5TrackJets.clone( rParam = 0.7 )
gk7TrackJets = gk5TrackJets.clone( rParam = 0.7 )
kt6TrackJets = kt4TrackJets.clone( rParam = 0.6 )
ca6TrackJets = ca4TrackJets.clone( rParam = 0.6 )


recoTrackJets   =cms.Sequence(trackWithVertexRefSelector+
                              trackRefsForJets+
                              ak5TrackJets+kt4TrackJets)

recoAllTrackJets=cms.Sequence(trackWithVertexRefSelector+
                              trackRefsForJets+
                              sisCone5TrackJets+sisCone7TrackJets+
                              kt4TrackJets+kt6TrackJets+
                              iterativeCone5TrackJets+
                              ak5TrackJets+ak7TrackJets+
                              gk5TrackJets+gk7TrackJets+
                              ca4TrackJets+ca6TrackJets)
