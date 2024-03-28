import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ZGammaplusJetsMonitor_cfi import hltZJetsmonitoring
from JetMETCorrections.Configuration.JetCorrectors_cff import *

#---- define trigger paths and module paths ---#
ZJet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/DimuonPFJet30',
        PathName = 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_PFJet30_v',
        ModuleName = 'hltDiMuon178Mass8Filtered',
        jets      = "ak4PFJetsPuppi",
        corrector = "ak4PFPuppiL1FastL2L3Corrector",
        # hlt muons       
        muonpt = 20.,
        muoneta = 2.3,
        # hlt Z cut
        Z_Dmass = 20.,
        Z_pt = 30.,
        # hlt jet
        ptcut = 30.,
        # back to back cut (between hlt Z and hlt jet)
        DeltaPhi = 2.7,
        # offline jet cut
        OfflineCut = 20.0,
        isMuonPath = True
)

Photon50Jet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/Photon50PFJet30',
        PathName = 'HLT_Photon50EB_TightID_TightIso_PFJet30_v',
        ModuleName = 'hltEG50EBTightIDTightIsoTrackIsoFilter',
        jets      = "ak4PFJetsPuppi",
        corrector = "ak4PFPuppiL1FastL2L3Corrector",
        # hlt jet
        ptcut = 30.,
        # back to back cut (between hlt photon and hlt jet)
        DeltaPhi = 2.7,
        # offline jet cut
        OfflineCut = 20.0,
        isMuonPath = False
)

Photon110Jet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/Photon110PFJet30',
        PathName = 'HLT_Photon110EB_TightID_TightIso_PFJet30_v',
        ModuleName = 'hltEG110EBTightIDTightIsoTrackIsoFilter',
        jets      = "ak4PFJetsPuppi",
        corrector = "ak4PFPuppiL1FastL2L3Corrector",
        # hlt jet
        ptcut = 30.,
        # back to back cut (between hlt photon and hlt jet)
        DeltaPhi = 2.7,
        # offline jet cut
        OfflineCut = 20.0,
        isMuonPath = False
)


HLTZGammaJetmonitoring = cms.Sequence(
    ak4PFPuppiL1FastL2L3CorrectorChain
    + ZJet_monitoring
    + Photon50Jet_monitoring
    + Photon110Jet_monitoring
    )
