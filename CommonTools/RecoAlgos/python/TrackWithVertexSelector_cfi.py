import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.TrackWithVertexSelectorParams_cff import *

trackWithVertexSelector = cms.EDFilter("TrackWithVertexSelector",
    trackWithVertexSelectorParams
)
