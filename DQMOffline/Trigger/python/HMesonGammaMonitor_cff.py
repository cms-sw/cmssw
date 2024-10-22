import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ObjMonitor_cfi import hltobjmonitoring

# HLT_
HMesonGammamonitoring = hltobjmonitoring.clone(
    FolderName = 'HLT/HIG/HMesonGamma/',
    phoSelection = "pt > 35 && abs(eta)<2.1 && hadTowOverEm<0.1 && full5x5_r9>0.9 && chargedHadronIso<1.295 && neutralHadronIso < 5.931+0.0163*pt+0.000014*pt*pt && photonIso < 6.641+0.0034*pt",
    trkSelection = "pt > 5 && quality('highPurity')",
    nphotons = 1,
    nmesons = 1,
    doMETHistos = False,
    doJetHistos = False,
    doHTHistos = False,
    doHMesonGammaHistos = True,
    #enableMETPlot = True,
    #metSelection = "pt>150",
    numGenericTriggerEventPSet = dict(hltInputTag   = ["TriggerResults","","HLT" ],
                                      hltPaths = ["HLT_Photon35_TwoProngs35_v*"])
)


hmesongammamonitoring = cms.Sequence(
    HMesonGammamonitoring
)

