import FWCore.ParameterSet.Config as cms

from RecoMuon.SimMuonAlgos.muonSimClassificationByHits_cff import *

muonrecosim = cms.Task(
    muonSimClassificationByHitsSequence
    )
