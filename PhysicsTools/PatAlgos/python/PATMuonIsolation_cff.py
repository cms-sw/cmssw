import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIsolationProducers.muIsolation_cff import *
patMuonIsolation = cms.Sequence(muIsolation)

