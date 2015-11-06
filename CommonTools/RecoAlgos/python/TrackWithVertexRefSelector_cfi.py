import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.TrackWithVertexSelectorParams_cff import *

trackWithVertexRefSelector = cms.EDProducer("TrackWithVertexRefSelector",
    trackWithVertexSelectorParams
)
