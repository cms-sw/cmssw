import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ZGammaplusJetsMonitor_cfi import hltZJetsmonitoring
from JetMETCorrections.Configuration.JetCorrectors_cff import *

## ========================================================================================= ##
ak8PFPuppiL1FastjetCorrector = ak4PFPuppiL1FastjetCorrector.clone()
ak8PFPuppiL2RelativeCorrector = ak4PFPuppiL2RelativeCorrector.clone(algorithm = 'AK8PFPuppi')
ak8PFPuppiL3AbsoluteCorrector =ak4PFPuppiL3AbsoluteCorrector.clone(algorithm = 'AK8PFPuppi')
ak8PFPuppiResidualCorrector = ak4PFPuppiResidualCorrector.clone(algorithm = 'AK8PFPuppi')
ak8PFPuppiL1FastL2L3ResidualCorrector = cms.EDProducer(
       'ChainedJetCorrectorProducer',
       correctors = cms.VInputTag('ak8PFPuppiL1FastjetCorrector','ak8PFPuppiL2RelativeCorrector', 'ak8PFPuppiL3AbsoluteCorrector', 'ak8PFPuppiResidualCorrector')
       )
ak8PFPuppiL1FastL2L3ResidualCorrectorTask = cms.Task(
	ak8PFPuppiL1FastjetCorrector, ak8PFPuppiL2RelativeCorrector, ak8PFPuppiL3AbsoluteCorrector, ak8PFPuppiResidualCorrector, ak8PFPuppiL1FastL2L3ResidualCorrector)
ak8PFPuppiL1FastL2L3ResidualCorrectorChain = cms.Sequence(ak8PFPuppiL1FastL2L3ResidualCorrectorTask)
## ========================================================================================= ##

#---- define trigger paths and module paths ---#
ZJet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/DimuonPFJet30',
        PathName = 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_PFJet30_v',
        ModuleName = 'hltDiMuon178Mass8Filtered',
        jets      = "ak4PFJetsPuppi",
        corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
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
        isMuonPath = True,
        # dr for matching
        dr2cut = 0.16
)

ZAK8Jet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/DimuonAK8PFJet30',
        PathName = 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_AK8PFJet30_v',
        ModuleName = 'hltDiMuon178Mass8Filtered',
        jets      = "ak8PFJetsPuppi",
        corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
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
        isMuonPath = True,
        # dr for matching
        dr2cut = 0.64
)


ZCaloJet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/DimuonCaloJet30',
        PathName = 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_CaloJet30_v',
        ModuleName = 'hltDiMuon178Mass8Filtered',
        jets      = "ak4CaloJets",
        corrector = "ak4CaloL1FastL2L3ResidualCorrector",
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
        isMuonPath = True,
        # dr for matching
        dr2cut = 0.16
)

ZAK8CaloJet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/DimuonAK8CaloJet30',
        PathName = 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_AK8CaloJet30_v',
        ModuleName = 'hltDiMuon178Mass8Filtered',
        jets      = "ak8PFJetsPuppi",
        corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
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
        isMuonPath = True,
        # dr for matching
        dr2cut = 0.64
)


Photon50Jet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/Photon50PFJet30',
        PathName = 'HLT_Photon50EB_TightID_TightIso_PFJet30_v',
        ModuleName = 'hltEG50EBTightIDTightIsoTrackIsoFilter',
        jets      = "ak4PFJetsPuppi",
        corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
        # hlt jet
        ptcut = 30.,
        # back to back cut (between hlt photon and hlt jet)
        DeltaPhi = 2.7,
        # offline jet cut
        OfflineCut = 20.0,
        isMuonPath = False,
        # dr for matching
        dr2cut = 0.16
)

Photon50AK8Jet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/Photon50AK8PFJet30',
        PathName = 'HLT_Photon50EB_TightID_TightIso_AK8PFJet30_v',
        ModuleName = 'hltEG50EBTightIDTightIsoTrackIsoFilter',
        jets      = "ak8PFJetsPuppi",
        corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
        # hlt jet
        ptcut = 30.,
        # back to back cut (between hlt photon and hlt jet)
        DeltaPhi = 2.7,
        # offline jet cut
        OfflineCut = 20.0,
        isMuonPath = False,
        # dr for matching
        dr2cut = 0.64
)

Photon50CaloJet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/Photon50CaloJet30',
        PathName = 'HLT_Photon50EB_TightID_TightIso_CaloJet30_v',
        ModuleName = 'hltEG50EBTightIDTightIsoTrackIsoFilter',
        jets      = "ak4CaloJets",
        corrector = "ak4CaloL1FastL2L3ResidualCorrector",
        # hlt jet
        ptcut = 30.,
        # back to back cut (between hlt photon and hlt jet)
        DeltaPhi = 2.7,
        # offline jet cut
        OfflineCut = 20.0,
        isMuonPath = False,
        # dr for matching
        dr2cut = 0.16
)

Photon50AK8CaloJet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/Photon50AK8CaloJet30',
        PathName = 'HLT_Photon50EB_TightID_TightIso_AK8CaloJet30_v',
        ModuleName = 'hltEG50EBTightIDTightIsoTrackIsoFilter',
        jets      = "ak8PFJetsPuppi",
        corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
        # hlt jet
        ptcut = 30.,
        # back to back cut (between hlt photon and hlt jet)
        DeltaPhi = 2.7,
        # offline jet cut
        OfflineCut = 20.0,
        isMuonPath = False,
        # dr for matching
        dr2cut = 0.64
)


Photon110Jet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/Photon110PFJet30',
        PathName = 'HLT_Photon110EB_TightID_TightIso_PFJet30_v',
        ModuleName = 'hltEG110EBTightIDTightIsoTrackIsoFilter',
        jets      = "ak4PFJetsPuppi",
        corrector = "ak4PFPuppiL1FastL2L3ResidualCorrector",
        # hlt jet
        ptcut = 30.,
        # back to back cut (between hlt photon and hlt jet)
        DeltaPhi = 2.7,
        # offline jet cut
        OfflineCut = 20.0,
        isMuonPath = False,
        # dr for matching
        dr2cut = 0.16
)

Photon110AK8Jet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/Photon110AK8PFJet30',
        PathName = 'HLT_Photon110EB_TightID_TightIso_AK8PFJet30_v',
        ModuleName = 'hltEG110EBTightIDTightIsoTrackIsoFilter',
        jets      = "ak8PFJetsPuppi",
        corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
        # hlt jet
        ptcut = 30.,
        # back to back cut (between hlt photon and hlt jet)
        DeltaPhi = 2.7,
        # offline jet cut
        OfflineCut = 20.0,
        isMuonPath = False,
        # dr for matching
        dr2cut = 0.64
)

Photon110CaloJet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/Photon110CaloJet30',
        PathName = 'HLT_Photon110EB_TightID_TightIso_CaloJet30_v',
        ModuleName = 'hltEG110EBTightIDTightIsoTrackIsoFilter',
        jets      = "ak4CaloJets",
        corrector = "ak4CaloL1FastL2L3ResidualCorrector",
        # hlt jet
        ptcut = 30.,
        # back to back cut (between hlt photon and hlt jet)
        DeltaPhi = 2.7,
        # offline jet cut
        OfflineCut = 20.0,
        isMuonPath = False,
        # dr for matching
        dr2cut = 0.16
)

Photon110AK8CaloJet_monitoring = hltZJetsmonitoring.clone(
        FolderName = 'HLT/JME/ZGammaPlusJets/Photon110Ak8CaloJet30',
        PathName = 'HLT_Photon110EB_TightID_TightIso_AK8CaloJet30_v',
        ModuleName = 'hltEG110EBTightIDTightIsoTrackIsoFilter',
        jets      = "ak8PFJetsPuppi",
        corrector = "ak8PFPuppiL1FastL2L3ResidualCorrector",
        # hlt jet
        ptcut = 30.,
        # back to back cut (between hlt photon and hlt jet)
        DeltaPhi = 2.7,
        # offline jet cut
        OfflineCut = 20.0,
        isMuonPath = False,
        # dr for matching
        dr2cut = 0.64
)



HLTZGammaJetmonitoring = cms.Sequence(
    ak4PFPuppiL1FastL2L3ResidualCorrectorChain
    + ak8PFPuppiL1FastL2L3ResidualCorrectorChain
    + ak4CaloL1FastL2L3ResidualCorrectorChain
    + ZJet_monitoring
    + ZAK8Jet_monitoring
    + ZCaloJet_monitoring
    + ZAK8CaloJet_monitoring
    + Photon50Jet_monitoring
    + Photon50AK8Jet_monitoring
    + Photon50CaloJet_monitoring
    + Photon50AK8CaloJet_monitoring
    + Photon110Jet_monitoring
    + Photon110AK8Jet_monitoring
    + Photon110CaloJet_monitoring
    + Photon110AK8CaloJet_monitoring
    )
