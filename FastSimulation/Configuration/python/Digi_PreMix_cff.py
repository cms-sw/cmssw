import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Digi_PreMix_cff import *

simMuonCSCDigis.InputCollection = 'MuonSimHitsMuonCSCHits'
simMuonDTDigis.InputCollection = 'MuonSimHitsMuonDTHits'
simMuonRPCDigis.InputCollection = 'MuonSimHitsMuonRPCHits'
calDigi.remove(castorDigiSequence)
