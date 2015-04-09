import FWCore.ParameterSet.Config as cms

# the actual digi sequence
from Configuration.StandardSequences.Digi_cff import *

simMuonCSCDigis.InputCollection = 'MuonSimHitsMuonCSCHits'
simMuonDTDigis.InputCollection = 'MuonSimHitsMuonDTHits'
simMuonRPCDigis.InputCollection = 'MuonSimHitsMuonRPCHits'
calDigi.remove(castorDigiSequence)

# give digi collections the names expected by RECO and HLT
import FastSimulation.Configuration.DigiAndMixAliasInfo_cff as _aliasInfo
generalTracks = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
ecalPreshowerDigis = _aliasInfo.infoToAlias(_aliasInfo.ecalPreShowerDigisAliasInfo)
ecalDigis = _aliasInfo.infoToAlias(_aliasInfo.ecalDigisAliasInfo)
hcalDigis = _aliasInfo.infoToAlias(_aliasInfo.hcalDigisAliasInfo)
muonDTDigis = _aliasInfo.infoToAlias(_aliasInfo.muonDTDigisAliasInfo)
muonCSCDigis = _aliasInfo.infoToAlias(_aliasInfo.muonCSCDigisAliasInfo)
muonRPCDigis = _aliasInfo.infoToAlias(_aliasInfo.muonRPCDigisAliasInfo)

