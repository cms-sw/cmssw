import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.photonjets_HLTPaths_cfi import *
photonjetsHLTFilter = cms.Sequence(singlePhotonHLTFilter+singleRelaxedPhotonHLTFilter+singlePhotonHLTFilter12)

