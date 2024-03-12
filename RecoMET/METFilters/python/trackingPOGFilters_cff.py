import FWCore.ParameterSet.Config as cms

from RecoMET.METFilters.trackingPOGFilters_cfi import *

trkPOGFilters = cms.Sequence( ~manystripclus53X * ~toomanystripclus53X * ~logErrorTooManyClusters )
# foo bar baz
# 5zs4leJsQ6TAV
# wP5bevmqZUiDZ
