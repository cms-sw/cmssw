import FWCore.ParameterSet.Config as cms

from METScanning.METdefaultTree.trackingPOGFilters_cfi import *

trkPOGFilters = cms.Sequence( ~manystripclus53X * ~toomanystripclus53X * ~logErrorTooManyClusters )
