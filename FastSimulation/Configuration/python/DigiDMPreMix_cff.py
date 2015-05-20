import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.DigiDMPreMix_cff import *

simMuonCSCDigis.InputCollection = 'MuonSimHitsMuonCSCHits'
simMuonDTDigis.InputCollection = 'MuonSimHitsMuonDTHits'
simMuonRPCDigis.InputCollection = 'MuonSimHitsMuonRPCHits'
#pdigi.remove(addPileupInfo)
