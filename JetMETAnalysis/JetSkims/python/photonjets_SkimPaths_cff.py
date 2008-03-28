import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.photonjets_Sequences_cff import *
singlePhotonHLTPath = cms.Path(singlePhotonHLTFilter)
singleRelaxedPhotonHLTPath = cms.Path(singleRelaxedPhotonHLTFilter)
singlePhotonHLTPath12 = cms.Path(singlePhotonHLTFilter12)

