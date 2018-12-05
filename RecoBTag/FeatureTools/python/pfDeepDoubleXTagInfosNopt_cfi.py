import FWCore.ParameterSet.Config as cms

from pfDeepDoubleXTagInfos_cfi import pfDeepDoubleXTagInfos

pfDeepDoubleXTagInfosNopt = pfDeepDoubleXTagInfos.clone(
  min_jet_pt = cms.double(0)
)
