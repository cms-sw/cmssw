from .adapt_to_new_backend import *
dqmitems={}

###########---- SMP selection goes here: #######################
### Made by hugo.ruben.delannoy@cern.ch, 03/23/2015          ###
### Updated by nicolas.jacques.s.postiau@cern.ch, 10/02/2017 ###
################################################################

def smpHLTlayouts(i, p, *rows):
    i["HLT/Layouts/" + p] = rows

## Collections for SMP
workDir = "SMP/"

#
## Summary plots
#

#to be done (not in a short-term plan)

#
## Single Muon: W or Z sample. MUO POG
#
paths = ["HLT_IsoMu24_eta2p1", "HLT_IsoMu27", "HLT_IsoMu20"]
#The Mu20 are maybe not necessary

#Instructions for the sample to use
smpHLTlayouts(dqmitems, workDir+"SingleMuon/00 - Instructions",
        [{'path': "Use ZMM samples", 'description':"Instructions for the paths in this folder"}])

for i, path in enumerate(paths):
    smpHLTlayouts(dqmitems, workDir+"SingleMuon/%02d - %s_gen" % (i+1, path),
            [{'path': "HLT/Muon/DistributionsGlobal/%s/efficiencyEta" % path, 'description':""}],
            [{'path': "HLT/Muon/DistributionsGlobal/%s/efficiencyPhi" % path, 'description':""}],
            [{'path': "HLT/Muon/DistributionsGlobal/%s/efficiencyTurnOn" % path, 'description':""}])

#
## Single Electron: W or Z sample. EGM POG
#
paths = ["HLT_Ele35_WPTight_Gsf", "HLT_Ele38_WPTight_Gsf", "HLT_Ele40_WPTight_Gsf", "HLT_Ele35_WPTight_Gsf_L1EGMT"]

#Instructions for the sample to use
smpHLTlayouts(dqmitems, workDir+"SingleElectron/00 - Instructions",
        [{'path': "Use ZEE samples", 'description':"Instructions for the paths in this folder"}])

smpHLTlayouts(dqmitems, workDir+"SingleElectron/01 - EffSummaryPaths_SingleEle_gen",
        [{'path': "HLT/SMP/SingleEle/EffSummaryPaths_SingleEle_gen", 'description':""}])
smpHLTlayouts(dqmitems, workDir+"SingleElectron/01 - EffSummaryPaths_SingleEle_rec",
        [{'path': "HLT/SMP/SingleEle/EffSummaryPaths_SingleEle_rec", 'description':""}])

for i, path in enumerate(paths):
    smpHLTlayouts(dqmitems, workDir+"SingleElectron/%02d - %s_gen" % (i+2, path),
            [{'path': "HLT/SMP/SingleEle/Eff_genEleEta_%s" % path, 'description':""}],
            [{'path': "HLT/SMP/SingleEle/Eff_genElePhi_%s" % path, 'description':""}],
            [{'path': "HLT/SMP/SingleEle/Eff_genEleMaxPt1_%s" % path, 'description':""}])
    smpHLTlayouts(dqmitems, workDir+"SingleElectron/%02d - %s_reco" % (i+2, path),
            [{'path': "HLT/SMP/SingleEle/Eff_recEleEta_%s" % path, 'description':""}],
            [{'path': "HLT/SMP/SingleEle/Eff_recElePhi_%s" % path, 'description':""}],
            [{'path': "HLT/SMP/SingleEle/Eff_recEleMaxPt1_%s" % path, 'description':""}])

#
## Double muon: Z sample. Higgs, HWW (didn't see these paths under MUO)
#
#not found in MUO, so we looked in HWW
paths = ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ"]

#Instructions for the sample to use
smpHLTlayouts(dqmitems, workDir+"DoubleMuon/00 - Instructions",
        [{'path': "Use ZMM samples", 'description':"Instructions for the paths in this folder"}])

for i, path in enumerate(paths):
    smpHLTlayouts(dqmitems, workDir+"DoubleMuon/%02d - %s_gen" % (i+1, path),
            [{'path': "HLT/Higgs/HWW/Eff_genMuEta_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/HWW/Eff_genMuPhi_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/HWW/Eff_genMuMaxPt1_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/HWW/Eff_genMuMaxPt2_%s" % path, 'description':""}])
    smpHLTlayouts(dqmitems, workDir+"DoubleMuon/%02d - %s_reco" % (i+1, path),
            [{'path': "HLT/Higgs/HWW/Eff_recMuEta_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/HWW/Eff_recMuPhi_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/HWW/Eff_recMuMaxPt1_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/HWW/Eff_recMuMaxPt2_%s" % path, 'description':""}])

#
## Double electron: Z sample. Higgs, HWW (didn't see any hists under EGM)
#
#not found in EGM, so we looked in HWW
path = "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"

#Instructions for the sample to use
smpHLTlayouts(dqmitems, workDir+"DoubleElectron/00 - Instructions",
        [{'path': "Use ZEE samples", 'description':"Instructions for the paths in this folder"}])

smpHLTlayouts(dqmitems, workDir+"DoubleElectron/%s_gen" % path,
        [{'path': "HLT/Higgs/HWW/Eff_genEleEta_%s" % path, 'description':""}],
        [{'path': "HLT/Higgs/HWW/Eff_genElePhi_%s" % path, 'description':""}],
        [{'path': "HLT/Higgs/HWW/Eff_genEleMaxPt1_%s" % path, 'description':""}],
        [{'path': "HLT/Higgs/HWW/Eff_genEleMaxPt2_%s" % path, 'description':""}])
smpHLTlayouts(dqmitems, workDir+"DoubleElectron/%s_reco" % path,
        [{'path': "HLT/Higgs/HWW/Eff_recEleEta_%s" % path, 'description':""}],
        [{'path': "HLT/Higgs/HWW/Eff_recElePhi_%s" % path, 'description':""}],
        [{'path': "HLT/Higgs/HWW/Eff_recEleMaxPt1_%s" % path, 'description':""}],
        [{'path': "HLT/Higgs/HWW/Eff_recEleMaxPt2_%s" % path, 'description':""}])

#
## MuEG: TTbarLepton sample. Higgs, HWW
#
paths = ["HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL", "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL"]

#Instructions for the sample to use
smpHLTlayouts(dqmitems, workDir+"MuEG/00 - Instructions",
        [{'path': "Use TTbarLepton samples", 'description':"Instructions for the paths in this folder"}])

for i, path in enumerate(paths):
    smpHLTlayouts(dqmitems, workDir+"MuEG/%02d - %s_gen" % (i+1, path),
            [{'path': "HLT/Higgs/HWW/Eff_genMuMaxPt1_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/HWW/Eff_genEleMaxPt1_%s" % path, 'description':""}])
    smpHLTlayouts(dqmitems, workDir+"MuEG/%02d - %s_reco" % (i+1, path),
            [{'path': "HLT/Higgs/HWW/Eff_recMuMaxPt1_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/HWW/Eff_recMuMaxPt2_%s" % path, 'description':""}])

#
## Single Photon: Hgg sample. SMP PAG
#

#Instructions for the sample to use
smpHLTlayouts(dqmitems, workDir+"SinglePhoton/00 - Instructions",
        [{'path': "Use H130GGgluonfusion samples", 'description':"Instructions for the paths in this folder"}])

smpHLTlayouts(dqmitems, workDir+"SinglePhoton/01 - EffSummaryPaths_SinglePhoton_gen",
        [{'path': "HLT/SMP/SinglePhoton/EffSummaryPaths_SinglePhoton_gen", 'description':""}])
smpHLTlayouts(dqmitems, workDir+"SinglePhoton/01 - EffSummaryPaths_SinglePhoton_rec",
        [{'path': "HLT/SMP/SinglePhoton/EffSummaryPaths_SinglePhoton_rec", 'description':""}])

numbers = ["33", "50", "75", "90", "120"] #maybe adding 165 but then I cannot loop like this since it's different for HLT_Photon et the other one
for number in numbers:
    paths = ["HLT_Photon"+number, "HLT_Photon"+number+"_R9Id90_HE10_IsoM"]

    for i, path in enumerate(paths):
        smpHLTlayouts(dqmitems, workDir+"SinglePhoton/%02d - %s_gen" % (i+2, path),
                [{'path': "HLT/SMP/SinglePhoton/Eff_genPhotonEta_%s" % path, 'description':""}],
                [{'path': "HLT/SMP/SinglePhoton/Eff_genPhotonPhi_%s" % path, 'description':""}],
                [{'path': "HLT/SMP/SinglePhoton/Eff_genPhotonMaxPt1_%s" % path, 'description':""}])
        smpHLTlayouts(dqmitems, workDir+"SinglePhoton/%02d - %s_reco" % (i+2, path),
                [{'path': "HLT/SMP/SinglePhoton/Eff_recPhotonEta_%s" % path, 'description':""}],
                [{'path': "HLT/SMP/SinglePhoton/Eff_recPhotonPhi_%s" % path, 'description':""}],
                [{'path': "HLT/SMP/SinglePhoton/Eff_recPhotonMaxPt1_%s" % path, 'description':""}])

#
## Double Photon: Hgg sample. Higgs/Hgg
#
paths = ["HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90", "HLT_Diphoton30EB_18EB_R9Id_OR_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55", "HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55"]

#Instructions for the sample to use
smpHLTlayouts(dqmitems, workDir+"DoublePhoton/00 - Instructions",
        [{'path': "Use H130GGgluonfusion samples", 'description':"Instructions for the paths in this folder"}])

for i, path in enumerate(paths):
    smpHLTlayouts(dqmitems, workDir+"DoublePhoton/%02d - %s_gen" % (i+1, path),
            [{'path': "HLT/Higgs/Hgg/Eff_genPhotonEta_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/Hgg/Eff_genPhotonPhi_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/Hgg/Eff_genPhotonMaxPt1_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/Hgg/Eff_genPhotonMaxPt2_%s" % path, 'description':""}])
    smpHLTlayouts(dqmitems, workDir+"DoublePhoton/%02d - %s_reco" % (i+1, path),
            [{'path': "HLT/Higgs/Hgg/Eff_recPhotonEta_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/Hgg/Eff_recPhotonPhi_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/Hgg/Eff_recPhotonMaxPt1_%s" % path, 'description':""}],
            [{'path': "HLT/Higgs/Hgg/Eff_recPhotonMaxPt2_%s" % path, 'description':""}])

#
## Single Jet: QCD sample(s), JetMET POG
#

#Instructions for the sample to use
smpHLTlayouts(dqmitems, workDir+"SingleJet/00 - Instructions",
        [{'path': "Use QCD_FlatPt_15_30HS samples", 'description':"Instructions for the paths in this folder"}])

smpHLTlayouts(dqmitems, workDir+"SingleJet/JetMET_TriggerRate",
        [{'path': "HLT/JetMET/TriggerSummary/JetMET_TriggerRate", 'description':""}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
