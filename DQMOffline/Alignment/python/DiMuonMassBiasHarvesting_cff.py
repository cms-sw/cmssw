import FWCore.ParameterSet.Config as cms

from DQMOffline.Alignment.DiMuonMassBiasClient_cfi import DiMuonMassBiasClient as diMuonMassBiasClient

# Z-> mm
__selectionName = 'TkAlDiMuonAndVertex'
ALCARECOTkAlZMuMuMassBiasClient = diMuonMassBiasClient.clone(
    FolderName = "AlCaReco/"+__selectionName
)

alcaTkAlZMuMuBiasClients = cms.Sequence(ALCARECOTkAlZMuMuMassBiasClient)

# J/psi -> mm
__selectionName = 'TkAlJpsiMuMu'
ALCARECOTkAlJpsiMuMuMassBiasClient = diMuonMassBiasClient.clone(
    FolderName = "AlCaReco/"+__selectionName,
    fitBackground = True,
    fit_par = dict(mean_par = [3.09, 2.7, 3.4],
                   width_par = [1.0, 0.0, 5.0],
                   sigma_par = [1.0, 0.0, 5.0])
)

alcaTkAlJpsiMuMuBiasClients = cms.Sequence(ALCARECOTkAlJpsiMuMuMassBiasClient)

# Upsilon -> mm
__selectionName = 'TkAlUpsilonMuMu'
ALCARECOTkAlUpsilonMuMuMassBiasClient = diMuonMassBiasClient.clone(
    FolderName = "AlCaReco/"+__selectionName,
    fitBackground = True,
    fit_par = dict(mean_par = [9.46, 8.9, 9.9],
                   width_par = [1.0, 0.0, 5.0],
                   sigma_par = [1.0, 0.0, 5.0])
)

alcaTkAlUpsilonMuMuBiasClients = cms.Sequence(ALCARECOTkAlUpsilonMuMuMassBiasClient)
