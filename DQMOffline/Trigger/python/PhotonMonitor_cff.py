import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.PhotonMonitor_cfi import hltPhotonmonitoring

#HLT_SinglePhoton200_IDTight
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

Photon60_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/EGM/Photon/Photon60/',
    photonSelection = "pt > 20 && r9() < 0.1 && ((eta<1.4442 && hadTowOverEm<0.0597 && full5x5_sigmaIetaIeta()<0.01031 && chargedHadronIso<1.295) || (eta<2.5 && eta>1.566 && hadTowOverEm<0.0481 && full5x5_sigmaIetaIeta()<0.03013 && chargedHadronIso<1.011))",
    denGenericTriggerEventPSet = dict(hltPaths = []),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon60_R9Id90_CaloIdL_IsoL_v*"])
)


Photon60_DisplacedIdL_monitoring = Photon60_monitoring.clone(
    FolderName = 'HLT/EXO/DisplacedPhoton/Photon60_DisplacedIdL/',
    denGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon60_R9Id90_CaloIdL_IsoL_v*"]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_v*"])
)


Photon60_DisplacedIdL_PFJet350MinPFJet15_monitoring = Photon60_DisplacedIdL_monitoring.clone(
    FolderName = 'HLT/EXO/DisplacedPhoton/Photon60_DisplacedIdL_PFJet350MinPFJet15/',
    denGenericTriggerEventPSet = dict(andOrHlt = False,
                                      hltPaths = ["HLT_Photon60_R9Id90_CaloIdL_IsoL_v*","HLT_PFHT350MinPFJet15_v*"]),
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Photon60_R9Id90_CaloIdL_IsoL_DisplacedIdL_PFHT350MinPFJet15_v*"])
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
    + Photon60_monitoring
    + Photon60_DisplacedIdL_monitoring
    + Photon60_DisplacedIdL_PFJet350MinPFJet15_monitoring
    + SinglePhoton50_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton75_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton90_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton120_R9Id90_HE10_IsoM_monitoring
    + SinglePhoton165_R9Id90_HE10_IsoM_monitoring
    + Photon50_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_PFMET50_monitoring
    + Photon75_R9Id90_HE10_IsoM_EBOnly_PFJetsMJJ300DEta3_monitoring
)


DiphotonMass90_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/HIG/DiPhoton/diphotonMass90/',
    nphotons = 2,
    photonSelection = "(pt > 20 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 20 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v*"])
)

DiphotonMass95_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/HIG/DiPhoton/diphotonMass95/',
    nphotons = 2,
    photonSelection = "(pt > 20 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 20 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v*"])
)
DiphotonMass55AND_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/HIG/DiPhoton/diphotonMass55AND/',
    nphotons = 2,
    photonSelection = "(pt > 20 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 20 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55_v*"]),
    histoPSet = dict(massBinning = [50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,69.,70.,75.,80.,90.,110.,150.])
)


DiphotonMass55EB_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/HIG/DiPhoton/diphotonMass55EB/',
    nphotons = 2,
    photonSelection = "(pt > 20 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30EB_18EB_R9Id_OR_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55_v*"]),
    histoPSet = dict(massBinning = [50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,69.,70.,75.,80.,90.,110.,150.])
)

DiphotonMass55ANDnoPV_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/HIG/DiPhoton/diphotonMass55ANDnoPV/',
    nphotons = 2,
    photonSelection = "(pt > 20 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 20 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55_v*"]),
    histoPSet = dict(massBinning = [50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,69.,70.,75.,80.,90.,110.,150.])
)


DiphotonMass55EBnoPV_monitoring = hltPhotonmonitoring.clone(
    FolderName = 'HLT/HIG/DiPhoton/diphotonMass55EBnoPV/',
    nphotons = 2,
    photonSelection = "(pt > 20 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30EB_18EB_R9Id_OR_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55_v*"]),
    histoPSet = dict(massBinning = [50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,69.,70.,75.,80.,90.,110.,150.])
)


DiphotonMass55NewAND_monitoring = hltPhotonmonitoring.clone(
    #FolderName = 'HLT/Photon/diphotonMass55NewAND/',
    FolderName = 'HLT/HIG/DiPhoton/diphotonMass55NewAND/',
    nphotons = 2,
    photonSelection = "(pt > 20 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 20 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30_18_PVrealAND_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55_v*"]),
    histoPSet = dict(massBinning = [50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,69.,70.,75.,80.,90.,110.,150.])
)


DiphotonMass55NewANDnoPV_monitoring = hltPhotonmonitoring.clone(
#DiphotonMass55NewANDnoPV_monitoring.FolderName = cms.string('HLT/Photon/diphotonMass55NewANDnoPV/')
    FolderName = 'HLT/HIG/DiPhoton/diphotonMass55NewANDnoPV/',
    nphotons = 2,
    photonSelection = "(pt > 20 && abs(eta)<1.4442 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.015 && full5x5_r9>.5)||(pt > 20 && abs(eta)<2.5 && abs(eta)>1.5556 && hadTowOverEm<0.12 && full5x5_sigmaIetaIeta()<0.035 && full5x5_r9>.8)",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_Diphoton30_18_PVrealAND_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55_v*"]),
    histoPSet = dict(massBinning = [50.,51.,52.,53.,54.,55.,56.,57.,58.,59.,60.,61.,62.,63.,64.,65.,66.,67.,68.,69.,70.,75.,80.,90.,110.,150.])
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
    +DiphotonMass55AND_monitoring
    +DiphotonMass55EB_monitoring
    +DiphotonMass55ANDnoPV_monitoring
    +DiphotonMass55EBnoPV_monitoring 
    +DiphotonMass55NewAND_monitoring
    +DiphotonMass55NewANDnoPV_monitoring
    +DiPhoton10Time1p4ns_monitoring
    +DiPhoton10sminlt0p1_monitoring
)
