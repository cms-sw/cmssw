import FWCore.ParameterSet.Config as cms

#setup names for multiple plots that use the same paths+modules
photon_pathName = "HLT_Photon30_R9Id90_HE10_IsoM"
photon_moduleName ="hltEG30R9Id90HE10IsoMTrackIsoFilter"

muon_pathName = "HLT_IsoMu27"
muon_moduleName = "hltL3crIsoL1sMu25L1f0L2f10QL3f27QL3trkIsoFiltered0p09"

l2muon_pathName = "HLT_L2Mu10"
l2muon_moduleName = "hltL2fL1sMu16L1f0L2Filtered10Q"

l2NoBPTXmuon_pathName = "HLT_L2Mu10_NoVertex_NoBPTX3BX_NoHalo"
l2NoBPTXmuon_moduleName = "hltL2fL1sMuOpenNotBptxORL1f0NoVtxCosmicSeedMeanTimerL2Filtered10"

electron_pathName = "HLT_Ele23_WPLoose_Gsf"
electron_moduleName = "hltEle23WPLooseGsfTrackIsoFilter"

caloMet_pathName = "HLT_MET60_IsoTrk35_Loose"
caloMet_moduleName = "hltMET60"

pfMet_pathName = "HLT_PFMET120_PFMHT120_IDTight"
pfMet_moduleName = "hltPFMET120"

jetAk8_pathName = "HLT_AK8PFJet360_TrimMass30"
jetAk8_moduleName = "hltAK8SinglePFJet360TrimModMass30"

rsq_mr_pathName = "HLT_RsqMR240_Rsq0p09_MR200"
rsq_mr_moduleName = "hltRsqMR240Rsq0p09MR200"

bJet_pathNameCalo = "HLT_PFMET120_BTagCSV_p067"
bJet_moduleNameCalo = "hltBTagCaloCSVp067Single"
bJet_pathNamePF = "HLT_QuadPFJet_BTagCSV_p016_VBF_Mqq500_v1"
bJet_moduleNamePF = "hltBTagPFCSVp016SingleWithMatching"

#To avoid booking histogram, set pathName = cms.string("")

hltObjectMonitor = cms.EDAnalyzer('HLTObjectMonitor',
    processName = cms.string("HLT"),
    alphaT = cms.PSet(
        pathName = cms.string("HLT_PFHT200_DiPFJetAve90_PFAlphaT0p63"),
        moduleName = cms.string("hltPFHT200PFAlphaT0p63"),
        axisLabel = cms.string("Alpha_{T}"),
        plotLabel = cms.string("alphaT"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(30),
        Xmin = cms.double(0),
        Xmax = cms.double(5)
        ),
    photonPt = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("Photon_pT"),
        axisLabel = cms.string("photon p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(100),
        Xmin = cms.double(0),
        Xmax = cms.double(200)
        ),
    photonEta = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("Photon_eta"),
        axisLabel = cms.string("photon eta"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3),
        Xmax = cms.double(3)
        ),
    photonPhi = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("Photon_phi"),
        axisLabel = cms.string("photon phi"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    muonPt = cms.PSet(
        pathName = cms.string(muon_pathName),
        moduleName = cms.string(muon_moduleName),
        plotLabel = cms.string("Muon_pT"),
        axisLabel = cms.string("muon p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(0),
        Xmax = cms.double(150)
        ),
    muonEta = cms.PSet(
        pathName = cms.string(muon_pathName),
        moduleName = cms.string(muon_moduleName),
        plotLabel = cms.string("Muon_eta"),
        axisLabel = cms.string("muon eta"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3),
        Xmax = cms.double(3)
        ),
    muonPhi = cms.PSet(
        pathName = cms.string(muon_pathName),
        moduleName = cms.string(muon_moduleName),
        plotLabel = cms.string("Muon_phi"),
        axisLabel = cms.string("muon phi"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    l2muonPt = cms.PSet(
        pathName = cms.string(l2muon_pathName),
        moduleName = cms.string(l2muon_moduleName),
        plotLabel = cms.string("L2Muon_pT"),
        axisLabel = cms.string("L2 muon p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(0),
        Xmax = cms.double(150)
        ),
    l2muonEta = cms.PSet(
        pathName = cms.string(l2muon_pathName),
        moduleName = cms.string(l2muon_moduleName),
        plotLabel = cms.string("L2Muon_eta"),
        axisLabel = cms.string("L2 muon eta"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3),
        Xmax = cms.double(3)
        ),
    l2muonPhi = cms.PSet(
        pathName = cms.string(l2muon_pathName),
        moduleName = cms.string(l2muon_moduleName),
        plotLabel = cms.string("L2Muon_phi"),
        axisLabel = cms.string("L2 muon phi"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    l2NoBPTXmuonPt = cms.PSet(
        pathName = cms.string(l2NoBPTXmuon_pathName),
        moduleName = cms.string(l2NoBPTXmuon_moduleName),
        plotLabel = cms.string("L2NoBPTXMuon_pT"),
        axisLabel = cms.string("L2 No BPTX muon p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(0),
        Xmax = cms.double(150)
        ),
    l2NoBPTXmuonEta = cms.PSet(
        pathName = cms.string(l2NoBPTXmuon_pathName),
        moduleName = cms.string(l2NoBPTXmuon_moduleName),
        plotLabel = cms.string("L2NoBPTXMuon_eta"),
        axisLabel = cms.string("L2 NoBPTX muon eta"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3),
        Xmax = cms.double(3)
        ),
    l2NoBPTXmuonPhi = cms.PSet(
        pathName = cms.string(l2NoBPTXmuon_pathName),
        moduleName = cms.string(l2NoBPTXmuon_moduleName),
        plotLabel = cms.string("L2NoBPTXMuon_phi"),
        axisLabel = cms.string("L2 NoBPTX muon phi"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    electronPt = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        plotLabel = cms.string("Electron_pT"),
        axisLabel = cms.string("electron p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(0),
        Xmax = cms.double(150)
        ),
    electronEta = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        plotLabel = cms.string("Electron_eta"),
        axisLabel = cms.string("electron eta"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3),
        Xmax = cms.double(3)
        ),
    electronPhi = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        plotLabel = cms.string("Electron_phi"),
        axisLabel = cms.string("electron phi"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    jetPt = cms.PSet(
        pathName = cms.string("HLT_PFJet200"),
        moduleName = cms.string("hltSinglePFJet200"),
        plotLabel = cms.string("Jet_pT"),
        axisLabel = cms.string("jet p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(150),
        Xmax = cms.double(550)
        ),
    jetAK8Pt = cms.PSet(
        pathName = cms.string(jetAk8_pathName),
        moduleName = cms.string(jetAk8_moduleName),
        axisLabel = cms.string("AK8 jet p_{T} [GeV]"),
        plotLabel = cms.string("JetAK8_Pt"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(300),
        Xmax = cms.double(750)
        ),
    jetAK8Mass = cms.PSet(
        pathName = cms.string(jetAk8_pathName),
        moduleName = cms.string(jetAk8_moduleName),
        plotLabel = cms.string("JetAK8_mass"),
        axisLabel = cms.string("AK8 jet mass [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(100),
        Xmin = cms.double(0),
        Xmax = cms.double(200)
        ),   
    tauPt = cms.PSet(
        pathName = cms.string("HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg"),
        moduleName = cms.string("hltDoublePFTau40TrackPt1MediumIsolationDz02Reg"),
        axisLabel = cms.string("tau p_{T} [GeV]"),
        plotLabel = cms.string("Tau_pT"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(30),
        Xmax = cms.double(350)
        ),
    diMuonLowMass = cms.PSet(
        pathName = cms.string("HLT_DoubleMu4_3_Jpsi_Displaced"),
        moduleName = cms.string("hltDisplacedmumuFilterDoubleMu43Jpsi"),
        plotLabel = cms.string("Dimuon_LowMass"),
        axisLabel = cms.string("di-muon low mass [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(100),
        Xmin = cms.double(2.5),
        Xmax = cms.double(3.5)
        ),
    caloMetPt = cms.PSet(
        pathName = cms.string(caloMet_pathName),
        moduleName = cms.string(caloMet_moduleName),
        plotLabel = cms.string("CaloMET_pT"),
        axisLabel = cms.string("calo MET p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(50),
        Xmax = cms.double(250)
        ),
    caloMetPhi = cms.PSet(
        pathName = cms.string(caloMet_pathName),
        moduleName = cms.string(caloMet_moduleName),
        plotLabel = cms.string("CaloMET_phi"),
        axisLabel = cms.string("calo MET phi"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    pfMetPt = cms.PSet(
        pathName = cms.string(pfMet_pathName),
        moduleName = cms.string(pfMet_moduleName),
        plotLabel = cms.string("PFMET_pT"),
        axisLabel = cms.string("PF MET p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(100),
        Xmax = cms.double(500)
        ),
    pfMetPhi = cms.PSet(
        pathName = cms.string(pfMet_pathName),
        moduleName = cms.string(pfMet_moduleName),
        plotLabel = cms.string("PFMET_phi"),
        axisLabel = cms.string("PF MET phi"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    caloHtPt = cms.PSet(
        pathName = cms.string("HLT_HT650_DisplacedDijet80_Inclusive"),
        moduleName = cms.string("hltHT650"),
        plotLabel = cms.string("CaloHT_pT"),
        axisLabel = cms.string("calo HT p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(200),
        Xmin = cms.double(0),
        Xmax = cms.double(2000)
        ),
    pfHtPt = cms.PSet(
        pathName = cms.string("HLT_PFHT750_4JetPt50"),
        moduleName = cms.string("hltPF4JetPt50HT750"),
        plotLabel = cms.string("PFHT_pT"),
        axisLabel = cms.string("PF HT p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(200),
        Xmin = cms.double(0),
        Xmax = cms.double(2000)
        ),
    bJetEta = cms.PSet(
        pathName = cms.string(bJet_pathNameCalo),
        moduleName = cms.string(bJet_moduleNameCalo),
        pathName_OR = cms.string(bJet_pathNamePF),
        moduleName_OR = cms.string(bJet_moduleNamePF),
        plotLabel = cms.string("bJet_eta"),
        axisLabel = cms.string("b-jet eta"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3),
        Xmax = cms.double(3)
        ),
    bJetPhi = cms.PSet(
        pathName = cms.string(bJet_pathNameCalo),
        moduleName = cms.string(bJet_moduleNameCalo),
        pathName_OR = cms.string(bJet_pathNamePF),
        moduleName_OR = cms.string(bJet_moduleNamePF),
        plotLabel = cms.string("bJet_phi"),
        axisLabel = cms.string("b-jet phi"),
        mainWorkspace = cms.bool(False),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    bJetCSVCalo = cms.PSet(
        pathName = cms.string(bJet_pathNameCalo),
        moduleName = cms.string(bJet_moduleNameCalo),
        plotLabel = cms.string("bJetCSVCalo"),
        axisLabel = cms.string("calo b-jet CSV"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(110),
        Xmin = cms.double(0),
        Xmax = cms.double(1)
        ),
    bJetCSVPF = cms.PSet(
        pathName = cms.string(bJet_pathNamePF),
        moduleName = cms.string(bJet_moduleNamePF),
        plotLabel = cms.string("bJetCSVPF"),
        axisLabel = cms.string("PF b-jet CSV"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(110),
        Xmin = cms.double(0),
        Xmax = cms.double(1)
        ),
    rsq = cms.PSet(
        pathName = cms.string(rsq_mr_pathName),
        moduleName = cms.string(rsq_mr_moduleName),
        plotLabel = cms.string("Rsq"),
        axisLabel = cms.string("R^{2}"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(30),
        Xmin = cms.double(0),
        Xmax = cms.double(2)
        ),                                  
    mr = cms.PSet(
        pathName = cms.string(rsq_mr_pathName),
        moduleName = cms.string(rsq_mr_moduleName),
        plotLabel = cms.string("mr"),
        axisLabel = cms.string("M_{R} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(2000)
        ),
    diMuonMass = cms.PSet(
        pathName = cms.string("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ"),
        moduleName = cms.string("hltDiMuonGlb17Glb8RelTrkIsoFiltered0p4DzFiltered0p2"),
        pathName_OR = cms.string("HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ"),
        moduleName_OR = cms.string("hltDiMuonGlb17Trk8RelTrkIsoFiltered0p4DzFiltered0p2"),
        plotLabel = cms.string("diMuon_Mass"),
        axisLabel = cms.string("dimuon mass [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(60),
        Xmax = cms.double(160)
        ),
    diElecMass = cms.PSet(
        pathName = cms.string("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"),
        moduleName = cms.string("hltEle23Ele12CaloIdLTrackIdLIsoVLDZFilter"),
        plotLabel = cms.string("di-Electron_Mass"),
        axisLabel = cms.string("di-electron mass [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(160)
        ),
    muonDxy = cms.PSet(
        pathName = cms.string("HLT_DoubleMu18NoFiltersNoVtx"),
        moduleName = cms.string("hltL3fDimuonL1f0L2NVf10L3NoFiltersNoVtxFiltered18"),
        plotLabel = cms.string("Muon_dxy"),
        axisLabel = cms.string("muon d_{xy} [mm]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(2000),
        Xmin = cms.double(-10),
        Xmax = cms.double(10)
        ),
    wallTime = cms.PSet(
        pathName = cms.string("wall time per event"),
        moduleName = cms.string(""),
        plotLabel = cms.string("wallTime"),
        axisLabel = cms.string("wall time per event [seconds]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(1000),
        Xmin = cms.double(0),
        Xmax = cms.double(0.005)
        )

)
