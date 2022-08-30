import FWCore.ParameterSet.Config as cms

from DQMOffline.Alignment.DiMuonMassBiasClient_cfi import DiMuonMassBiasClient as diMuonMassBiasClient

__selectionName = 'TkAlDiMuonAndVertex'
ALCARECOTkAlZMuMuMassBiasClient = diMuonMassBiasClient.clone(
    FolderName = "AlCaReco/"+__selectionName
)

alcaTkAlZMuMuBiasClients = cms.Sequence(ALCARECOTkAlZMuMuMassBiasClient)
