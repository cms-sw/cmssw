import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.PhotonMonitor_cfi import hltPhotonmonitoring

#HLT_SinglePhoton300_IDTight
SinglePhoton300_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EGM/Photon/Photon300/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon300_NoHE_v*"])
)


# HLT_SinglePhoton200_IDTight
SinglePhoton200_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EGM/Photon/Photon200/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon200_v*"])
)


SinglePhoton50_R9Id90_HE10_IsoM_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EGM/Photon/Photon50_R9Id90_HE10_IsoM/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon50_R9Id90_HE10_IsoM_v*"])
)


SinglePhoton75_R9Id90_HE10_IsoM_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EGM/Photon/Photon75_R9Id90_HE10_IsoM/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon75_R9Id90_HE10_IsoM_v*"])
)


SinglePhoton90_R9Id90_HE10_IsoM_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EGM/Photon/Photon90_R9Id90_HE10_IsoM/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon90_R9Id90_HE10_IsoM_v*"])
)

SinglePhoton120_R9Id90_HE10_IsoM_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EGM/Photon/Photon120_R9Id90_HE10_IsoM/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon120_R9Id90_HE10_IsoM_v*"])
)

SinglePhoton165_R9Id90_HE10_IsoM_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EGM/Photon/Photon165_R9Id90_HE10_IsoM/',
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon165_R9Id90_HE10_IsoM_v*"])
)

Photon60_DisplacedIdL_PFHT350_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EXO/DisplacedPhoton/Photon60_DisplacedIdL_PFHT350/',
    photonSelection = "pt > 20 && r9() < 0.1 && ((eta<1.4442 && hadTowOverEm<0.0597 && full5x5_sigmaIetaIeta()<0.01031 && chargedHadronIso<1.295) || (eta<2.5 && eta>1.566 && hadTowOverEm<0.0481 && full5x5_sigmaIetaIeta()<0.03013 && chargedHadronIso<1.011))",
    denGenericTriggerEventPSet = dict(andOrHlt = False,
                                      hltPaths = ["HLT_Photon50_R9Id90_HE10_IsoM_v*","HLT_PFHT350_v*"]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350_v*"])
)


from DQMOffline.Trigger.ObjMonitor_cfi import hltobjmonitoring

Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50_monitoring = hltobjmonitoring.clone(
    #FolderName = 'HLT/Photon/Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50/',
    FolderName = 'HLT/EXO/Photon/Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50/',
    denGenericTriggerEventPSet = hltobjmonitoring.numGenericTriggerEventPSet.clone(
        hltPaths = ["HLT_Photon50_R9Id90_HE10_IsoM_v*"]
    ),
    numGenericTriggerEventPSet = hltobjmonitoring.numGenericTriggerEventPSet.clone(
        hltPaths = ["HLT_Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50_v*"]
    ),
    phoSelection = 'pt > 80 & abs(eta) < 1.44',
    nphotons = 1,
    jetSelection = "pt > 30 & abs(eta) < 5.0",
    jetId = "tight",
    njets = 2,
    doHTHistos = False,
    histoPSet = dict(
        mjjBinning = [20. * x for x in range(30)],
        metPSet = dict(
                nbins = 20,
                xmin = -0.5,
                xmax = 200.)
        )
)

Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_monitoring = hltobjmonitoring.clone(
#    FolderName = 'HLT/Photon/Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3/',
    FolderName = 'HLT/EXO/Photon/Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3/',
    denGenericTriggerEventPSet = hltobjmonitoring.numGenericTriggerEventPSet.clone(
        hltPaths = ["HLT_Photon75_R9Id90_HE10_IsoM_v*"]
    ),
    numGenericTriggerEventPSet = hltobjmonitoring.numGenericTriggerEventPSet.clone(
        hltPaths = ["HLT_Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_v*"]
    ),
    phoSelection = 'pt > 80 & abs(eta) < 1.44',
    nphotons = 1,
    jetSelection = "pt > 30 & abs(eta) < 5.0",
    jetId = "tight",
    njets = 2,
    doMETHistos = False,
    doHTHistos = False,
    histoPSet = dict(mjjBinning = [20. * x for x in range(30)])
)

exoHLTPhotonmonitoring = cms.Sequence(
    SinglePhoton300_monitoring
    + SinglePhoton200_monitoring
    + Photon60_DisplacedIdL_PFHT350_monitoring
    + SinglePhoton50_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton75_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton90_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton120_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton165_R9Id90_HE10_IsoM_monitoring
    + Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50_monitoring
    + Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_monitoring
)


DiphotonMass90_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/HIG/DiPhoton/diphoton3022Mass90/',
    nphotons = 2,
    photonSelection = "(pt > 20 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 20 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v*"])
)

DiphotonMass95_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/HIG/DiPhoton/diphoton3022Mass95/',
    nphotons = 2,
    photonSelection = "(pt > 20 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 20 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v*"])
)

DiphotonMass55_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/HIG/DiPhoton/diphoton3018Mass55/',
    nphotons = 2,
    photonSelection = "(pt > 15 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 15 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_Mass55_v*"]),
    histoPSet = dict(massBinning = [50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,69.,70.,75.,80.,90.,110.,150.])
)

Diphoton3018_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EXO/DiPhoton/DiPhoton3018/',
    nphotons = 2,
    photonSelection = "(pt > 15 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 15 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_v*"]),
    histoPSet = dict(massBinning = [10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,90.,110.,150.])
)

Diphoton2214_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EXO/DiPhoton/DiPhoton2214/',
    nphotons = 2,
    photonSelection = "(pt > 10 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 10 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton22_14_eta1p5_R9IdL_AND_HET_AND_IsoTCaloIdT_v*"]),
    histoPSet = dict(massBinning = [10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,90.,110.,150.])
)


DiPhoton10Time1p4ns_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EXO/DiPhoton/DiPhoton10Time1p4ns/',
    nphotons = 2,
    photonSelection = "(pt > 10 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 10 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DiPhoton10Time1p4ns_v*"]),
)

DiPhoton10sminlt0p1_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EXO/DiPhoton/DiPhoton10sminlt0p1/',
    nphotons = 2,
    photonSelection = "(pt > 10 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 10 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_DiPhoton10sminlt0p1_v*"]),
)

higgsHLTDiphotonMonitoring = cms.Sequence(
    DiphotonMass90_monitoring
    +DiphotonMass95_monitoring
    +DiphotonMass55_monitoring
    +Diphoton3018_monitoring
    +Diphoton2214_monitoring
    +DiPhoton10Time1p4ns_monitoring
    +DiPhoton10sminlt0p1_monitoring
)
