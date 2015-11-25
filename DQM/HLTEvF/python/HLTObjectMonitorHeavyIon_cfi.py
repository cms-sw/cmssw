import FWCore.ParameterSet.Config as cms

#setup names for multiple plots that use the same paths+modules
caloJet_HI_pathName = "HLT_PuAK4CaloJet60_Eta5p1"
caloJet_HI_moduleName = "hltSinglePuAK4CaloJet60Eta5p1"

singlePhoton20Eta1p5_HI_pathName = "HLT_HISinglePhoton20_Eta1p5"
singlePhoton20Eta1p5_HI_moduleName = "hltHIPhoton20Eta1p5"

singlePhoton20Eta3p1_HI_pathName = "HLT_HISinglePhoton20_Eta3p1"
singlePhoton20Eta3p1_HI_moduleName = "hltHIPhoton20Eta3p1"

fullTrack12_MinBiasHF_HI_pathName = "HLT_HIFullTrack12_L1MinimumBiasHF1_AND"
fullTrack12_Centrality010_HI_pathName = "HLT_HIFullTrack12_L1Centrality010"
fullTrack12_Centrality301_HI_pathName = "HLT_HIFullTrack12_L1Centrality30100"
fullTrack12_HI_moduleName = "hltHIFullTrackFilter12"

fullTrack24_HI_pathName = "HLT_HIFullTrack24"
fullTrack24_Centrality301_HI_pathName = "HLT_HIFullTrack24_L1Centrality30100"
fullTrack24_HI_moduleName = "hltHIFullTrackFilter24"

fullTrack34_HI_pathName = "HLT_HIFullTrack34"
fullTrack34_Centrality301_HI_pathName = "HLT_HIFullTrack34_L1Centrality30100"
fullTrack34_HI_moduleName = "hltHIFullTrackFilter34"

dMeson_HI_pathName = "HLT_DmesonHITrackingGlobal_Dpt40"
#dMeson_HI_moduleName = "HLTtktkFilterForDmesonDp40"
dMeson_HI_moduleName = "HLTtktkFilterForDmesonGlobal8Dp40"

l1DoubleMu_HI_pathName = "HLT_HIL1DoubleMu0_Cent30"
l1DoubleMu_HI_moduleName = "hltHIDoubleMu0MinBiasCent30L1Filtered"

l3DoubleMu_HI_pathName = "HLT_HIL3DoubleMu0_Cent30"
l3DoubleMu_HI_moduleName = "hltHIDimuonOpenCentrality30L3Filter"

l2SingleMu_HI_pathName = "HLT_HIL2Mu15"
l2SingleMu_HI_moduleName = "hltHIL2Mu15L2Filtered"

l3SingleMu_HI_pathName = "HLT_HIL3Mu15"
l3SingleMu_HI_moduleName = "hltHISingleMu15L3Filtered"

l3SingleMuHF_HI_pathName = "HLT_HIL3Mu15_2HF"
l3SingleMuHF_HI_moduleName = "hltHISingleMu152HFL3Filtered"

#pp ref paths
caloJetRef_HI_pathName = "HLT_AK4CaloJet60_Eta5p1"
caloJetRef_HI_moduleName = "hltSingleAK4CaloJet60Eta5p1"

dMesonRef_HI_pathName = "HLT_DmesonPPTrackingGlobal_Dpt40"
dMesonRef_HI_moduleName = "HLTtktkFilterForDmesonDp40"

#To avoid booking histogram, set pathName = cms.string("")

hltObjectMonitorHeavyIon = cms.EDAnalyzer('HLTObjectMonitorHeavyIon',
    processName = cms.string("HLT"),
    wallTime = cms.PSet(
        pathName = cms.string("wall time per event"),
        moduleName = cms.string(""),
        plotLabel = cms.string("wallTime"),
        axisLabel = cms.string("wall time per event [seconds]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(1000),
        Xmin = cms.double(0),
        Xmax = cms.double(0.005)
        ),
    caloJetPt_HI = cms.PSet(
        pathName = cms.string(caloJet_HI_pathName),
        moduleName = cms.string(caloJet_HI_moduleName),
        plotLabel = cms.string("caloJet_pt_HI"),
        axisLabel = cms.string("caloJet pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(50),
        Xmax = cms.double(100)
        ),
    caloJetEta_HI = cms.PSet(
        pathName = cms.string(caloJet_HI_pathName),
        moduleName = cms.string(caloJet_HI_moduleName),
        plotLabel = cms.string("caloJet_eta_HI"),
        axisLabel = cms.string("caloJet eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-5),
        Xmax = cms.double(5)
        ),
    caloJetPhi_HI = cms.PSet(
        pathName = cms.string(caloJet_HI_pathName),
        moduleName = cms.string(caloJet_HI_moduleName),
        plotLabel = cms.string("caloJet_phi_HI"),
        axisLabel = cms.string("caloJet phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    singlePhotonEta1p5Pt_HI = cms.PSet(
        pathName = cms.string(singlePhoton20Eta1p5_HI_pathName),
        moduleName = cms.string(singlePhoton20Eta1p5_HI_moduleName),
        plotLabel = cms.string("singlePhoton20Eta1p5_pt_HI"),
        axisLabel = cms.string("photon pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(15),
        Xmax = cms.double(100)
        ),
    singlePhotonEta1p5Eta_HI = cms.PSet(
        pathName = cms.string(singlePhoton20Eta1p5_HI_pathName),
        moduleName = cms.string(singlePhoton20Eta1p5_HI_moduleName),
        plotLabel = cms.string("singlePhoton20Eta1p5_eta_HI"),
        axisLabel = cms.string("photon eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-1.6),
        Xmax = cms.double(1.6)
        ),
    singlePhotonEta1p5Phi_HI = cms.PSet(
        pathName = cms.string(singlePhoton20Eta1p5_HI_pathName),
        moduleName = cms.string(singlePhoton20Eta1p5_HI_moduleName),
        plotLabel = cms.string("singlePhoton20Eta1p5_phi_HI"),
        axisLabel = cms.string("photon phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    singlePhotonEta3p1Pt_HI = cms.PSet(
        pathName = cms.string(singlePhoton20Eta3p1_HI_pathName),
        moduleName = cms.string(singlePhoton20Eta3p1_HI_moduleName),
        plotLabel = cms.string("singlePhoton20Eta3p1_pt_HI"),
        axisLabel = cms.string("photon pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(15),
        Xmax = cms.double(100)
        ),
    singlePhotonEta3p1Eta_HI = cms.PSet(
        pathName = cms.string(singlePhoton20Eta3p1_HI_pathName),
        moduleName = cms.string(singlePhoton20Eta3p1_HI_moduleName),
        plotLabel = cms.string("singlePhoton20Eta3p1_eta_HI"),
        axisLabel = cms.string("photon eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    singlePhotonEta3p1Phi_HI = cms.PSet(
        pathName = cms.string(singlePhoton20Eta3p1_HI_pathName),
        moduleName = cms.string(singlePhoton20Eta3p1_HI_moduleName),
        plotLabel = cms.string("singlePhoton20Eta3p1_phi_HI"),
        axisLabel = cms.string("photon phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    doublePhotonMass_HI = cms.PSet(
        pathName = cms.string("HLT_HIDoublePhoton15_Eta1p5_Mass50_1000"),
        moduleName = cms.string("hltHIDoublePhoton15Eta1p5Mass501000Filter"),
        plotLabel = cms.string("doublePhoton_mass_HI"),
        axisLabel = cms.string("photon mass [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(40),
        Xmax = cms.double(140)
        ),
    fullTrack12MinBiasHFPt_HI = cms.PSet(
        pathName = cms.string(fullTrack12_MinBiasHF_HI_pathName),
        moduleName = cms.string(fullTrack12_HI_moduleName),
        plotLabel = cms.string("fullTrack12MinBiasHF_pt_HI"),
        axisLabel = cms.string("track pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(300)
        ),
    fullTrack12MinBiasHFEta_HI = cms.PSet(
        pathName = cms.string(fullTrack12_MinBiasHF_HI_pathName),
        moduleName = cms.string(fullTrack12_HI_moduleName),
        plotLabel = cms.string("fullTrack12MinBiasHF_eta_HI"),
        axisLabel = cms.string("track eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    fullTrack12MinBiasHFPhi_HI = cms.PSet(
        pathName = cms.string(fullTrack12_MinBiasHF_HI_pathName),
        moduleName = cms.string(fullTrack12_HI_moduleName),
        plotLabel = cms.string("fullTrack12MinBiasHF_phi_HI"),
        axisLabel = cms.string("track phi "),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    fullTrack12Centrality010Pt_HI = cms.PSet(
        pathName = cms.string(fullTrack12_Centrality010_HI_pathName),
        moduleName = cms.string(fullTrack12_HI_moduleName),
        plotLabel = cms.string("fullTrack12Centrality010_pt_HI"),
        axisLabel = cms.string("track pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(300)
        ),
    fullTrack12Centrality010Eta_HI = cms.PSet(
        pathName = cms.string(fullTrack12_Centrality010_HI_pathName),
        moduleName = cms.string(fullTrack12_HI_moduleName),
        plotLabel = cms.string("fullTrack12Centrality010_eta_HI"),
        axisLabel = cms.string("track eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    fullTrack12Centrality010Phi_HI = cms.PSet(
        pathName = cms.string(fullTrack12_Centrality010_HI_pathName),
        moduleName = cms.string(fullTrack12_HI_moduleName),
        plotLabel = cms.string("fullTrack12Centrality010_phi_HI"),
        axisLabel = cms.string("track phi "),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    fullTrack12Centrality301Pt_HI = cms.PSet(
        pathName = cms.string(fullTrack12_Centrality301_HI_pathName),
        moduleName = cms.string(fullTrack12_HI_moduleName),
        plotLabel = cms.string("fullTrack12Centrality301_pt_HI"),
        axisLabel = cms.string("track pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(300)
        ),
    fullTrack12Centrality301Eta_HI = cms.PSet(
        pathName = cms.string(fullTrack12_Centrality301_HI_pathName),
        moduleName = cms.string(fullTrack12_HI_moduleName),
        plotLabel = cms.string("fullTrack12Centrality301_eta_HI"),
        axisLabel = cms.string("track eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    fullTrack12Centrality301Phi_HI = cms.PSet(
        pathName = cms.string(fullTrack12_Centrality301_HI_pathName),
        moduleName = cms.string(fullTrack12_HI_moduleName),
        plotLabel = cms.string("fullTrack12Centrality301_phi_HI"),
        axisLabel = cms.string("track phi "),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    fullTrack24Pt_HI = cms.PSet(
        pathName = cms.string(fullTrack24_HI_pathName),
        moduleName = cms.string(fullTrack24_HI_moduleName),
        plotLabel = cms.string("fullTrack24_pt_HI"),
        axisLabel = cms.string("track pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(300)
        ),
    fullTrack24Eta_HI = cms.PSet(
        pathName = cms.string(fullTrack24_HI_pathName),
        moduleName = cms.string(fullTrack24_HI_moduleName),
        plotLabel = cms.string("fullTrack24_eta_HI"),
        axisLabel = cms.string("track eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    fullTrack24Phi_HI = cms.PSet(
        pathName = cms.string(fullTrack24_HI_pathName),
        moduleName = cms.string(fullTrack24_HI_moduleName),
        plotLabel = cms.string("fullTrack24_phi_HI"),
        axisLabel = cms.string("track phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    fullTrack24Centrality301Pt_HI = cms.PSet(
        pathName = cms.string(fullTrack24_Centrality301_HI_pathName),
        moduleName = cms.string(fullTrack24_HI_moduleName),
        plotLabel = cms.string("fullTrack24Centrality301_pt_HI"),
        axisLabel = cms.string("track pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(300)
        ),
    fullTrack24Centrality301Eta_HI = cms.PSet(
        pathName = cms.string(fullTrack24_Centrality301_HI_pathName),
        moduleName = cms.string(fullTrack24_HI_moduleName),
        plotLabel = cms.string("fullTrack24Centrality301_eta_HI"),
        axisLabel = cms.string("track eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    fullTrack24Centrality301Phi_HI = cms.PSet(
        pathName = cms.string(fullTrack24_Centrality301_HI_pathName),
        moduleName = cms.string(fullTrack24_HI_moduleName),
        plotLabel = cms.string("fullTrack24Centrality301_phi_HI"),
        axisLabel = cms.string("track phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    fullTrack34Pt_HI = cms.PSet(
        pathName = cms.string(fullTrack34_HI_pathName),
        moduleName = cms.string(fullTrack34_HI_moduleName),
        plotLabel = cms.string("fullTrack34_pt_HI"),
        axisLabel = cms.string("track pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(300)
        ),
    fullTrack34Eta_HI = cms.PSet(
        pathName = cms.string(fullTrack34_HI_pathName),
        moduleName = cms.string(fullTrack34_HI_moduleName),
        plotLabel = cms.string("fullTrack34_eta_HI"),
        axisLabel = cms.string("track eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    fullTrack34Phi_HI = cms.PSet(
        pathName = cms.string(fullTrack34_HI_pathName),
        moduleName = cms.string(fullTrack34_HI_moduleName),
        plotLabel = cms.string("fullTrack34_phi_HI"),
        axisLabel = cms.string("track phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    fullTrack34Centrality301Pt_HI = cms.PSet(
        pathName = cms.string(fullTrack34_Centrality301_HI_pathName),
        moduleName = cms.string(fullTrack34_HI_moduleName),
        plotLabel = cms.string("fullTrack34Centrality301_pt_HI"),
        axisLabel = cms.string("track pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(300)
        ),
    fullTrack34Centrality301Eta_HI = cms.PSet(
        pathName = cms.string(fullTrack34_Centrality301_HI_pathName),
        moduleName = cms.string(fullTrack34_HI_moduleName),
        plotLabel = cms.string("fullTrack34Centrality301_eta_HI"),
        axisLabel = cms.string("track eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    fullTrack34Centrality301Phi_HI = cms.PSet(
        pathName = cms.string(fullTrack34_Centrality301_HI_pathName),
        moduleName = cms.string(fullTrack34_HI_moduleName),
        plotLabel = cms.string("fullTrack34Centrality301_phi_HI"),
        axisLabel = cms.string("track phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    dMesonTrackPt_HI = cms.PSet(
        pathName = cms.string(dMeson_HI_pathName),
        moduleName = cms.string(dMeson_HI_moduleName),
        plotLabel = cms.string("dMesonTrack_pt_HI"),
        axisLabel = cms.string("track pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(60)
        ),
    dMesonTrackEta_HI = cms.PSet(
        pathName = cms.string(dMeson_HI_pathName),
        moduleName = cms.string(dMeson_HI_moduleName),
        plotLabel = cms.string("dMesonTrack_eta_HI"),
        axisLabel = cms.string("track eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    dMesonTrackSystemMass_HI = cms.PSet(
        pathName = cms.string(dMeson_HI_pathName),
        moduleName = cms.string(dMeson_HI_moduleName),
        plotLabel = cms.string("dMesonTrack_systemMass_HI"),
        axisLabel = cms.string("inv mass 2 track system [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(1.34),
        Xmax = cms.double(2.6)
        ),
    dMesonTrackSystemPt_HI = cms.PSet(
        pathName = cms.string(dMeson_HI_pathName),
        moduleName = cms.string(dMeson_HI_moduleName),
        plotLabel = cms.string("dMesonTrack_systemPt_HI"),
        axisLabel = cms.string("pt 2 track system [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(5),
        Xmax = cms.double(100)
        ),
    l1DoubleMuPt_HI = cms.PSet(
        pathName = cms.string(l1DoubleMu_HI_pathName),
        moduleName = cms.string(l1DoubleMu_HI_moduleName),
        plotLabel = cms.string("l1DoubleMu_pt_HI"),
        axisLabel = cms.string("pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(400)
        ),
    l1DoubleMuEta_HI = cms.PSet(
        pathName = cms.string(l1DoubleMu_HI_pathName),
        moduleName = cms.string(l1DoubleMu_HI_moduleName),
        plotLabel = cms.string("l1DoubleMu_eta_HI"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    l1DoubleMuPhi_HI = cms.PSet(
        pathName = cms.string(l1DoubleMu_HI_pathName),
        moduleName = cms.string(l1DoubleMu_HI_moduleName),
        plotLabel = cms.string("l1DoubleMu_phi_HI"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    l3DoubleMuPt_HI = cms.PSet(
        pathName = cms.string(l3DoubleMu_HI_pathName),
        moduleName = cms.string(l3DoubleMu_HI_moduleName),
        plotLabel = cms.string("l3DoubleMu_pt_HI"),
        axisLabel = cms.string("pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(400)
        ),
    l3DoubleMuEta_HI = cms.PSet(
        pathName = cms.string(l3DoubleMu_HI_pathName),
        moduleName = cms.string(l3DoubleMu_HI_moduleName),
        plotLabel = cms.string("l3DoubleMu_eta_HI"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    l3DoubleMuPhi_HI = cms.PSet(
        pathName = cms.string(l3DoubleMu_HI_pathName),
        moduleName = cms.string(l3DoubleMu_HI_moduleName),
        plotLabel = cms.string("l3DoubleMu_phi_HI"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    l2SingleMuPt_HI = cms.PSet(
        pathName = cms.string(l2SingleMu_HI_pathName),
        moduleName = cms.string(l2SingleMu_HI_moduleName),
        plotLabel = cms.string("l2SingleMu_pt_HI"),
        axisLabel = cms.string("pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(400)
        ),
    l2SingleMuEta_HI = cms.PSet(
        pathName = cms.string(l2SingleMu_HI_pathName),
        moduleName = cms.string(l2SingleMu_HI_moduleName),
        plotLabel = cms.string("l2SingleMu_eta_HI"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    l2SingleMuPhi_HI = cms.PSet(
        pathName = cms.string(l2SingleMu_HI_pathName),
        moduleName = cms.string(l2SingleMu_HI_moduleName),
        plotLabel = cms.string("l2SingleMu_phi_HI"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    l3SingleMuPt_HI = cms.PSet(
        pathName = cms.string(l3SingleMu_HI_pathName),
        moduleName = cms.string(l3SingleMu_HI_moduleName),
        plotLabel = cms.string("l3SingleMu_pt_HI"),
        axisLabel = cms.string("pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(400)
        ),
    l3SingleMuEta_HI = cms.PSet(
        pathName = cms.string(l3SingleMu_HI_pathName),
        moduleName = cms.string(l3SingleMu_HI_moduleName),
        plotLabel = cms.string("l3SingleMu_eta_HI"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    l3SingleMuPhi_HI = cms.PSet(
        pathName = cms.string(l3SingleMu_HI_pathName),
        moduleName = cms.string(l3SingleMu_HI_moduleName),
        plotLabel = cms.string("l3SingleMu_phi_HI"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    l3SingleMuHFPt_HI = cms.PSet(
        pathName = cms.string(l3SingleMuHF_HI_pathName),
        moduleName = cms.string(l3SingleMuHF_HI_moduleName),
        plotLabel = cms.string("l3SingleMuHF_pt_HI"),
        axisLabel = cms.string("pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(400)
        ),
    l3SingleMuHFEta_HI = cms.PSet(
        pathName = cms.string(l3SingleMuHF_HI_pathName),
        moduleName = cms.string(l3SingleMuHF_HI_moduleName),
        plotLabel = cms.string("l3SingleMuHF_eta_HI"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    l3SingleMuHFPhi_HI = cms.PSet(
        pathName = cms.string(l3SingleMuHF_HI_pathName),
        moduleName = cms.string(l3SingleMuHF_HI_moduleName),
        plotLabel = cms.string("l3SingleMuHF_phi_HI"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
#pp ref                                  
    caloJetRefPt_HI = cms.PSet(
        pathName = cms.string(caloJetRef_HI_pathName),
        moduleName = cms.string(caloJetRef_HI_moduleName),
        plotLabel = cms.string("caloJetRef_pt_HI"),
        axisLabel = cms.string("caloJetRef pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(50),
        Xmax = cms.double(150)
        ),
    caloJetRefEta_HI = cms.PSet(
        pathName = cms.string(caloJetRef_HI_pathName),
        moduleName = cms.string(caloJetRef_HI_moduleName),
        plotLabel = cms.string("caloJetRef_eta_HI"),
        axisLabel = cms.string("caloJetRef eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-5),
        Xmax = cms.double(5)
        ),
    caloJetRefPhi_HI = cms.PSet(
        pathName = cms.string(caloJetRef_HI_pathName),
        moduleName = cms.string(caloJetRef_HI_moduleName),
        plotLabel = cms.string("caloJetRef_phi_HI"),
        axisLabel = cms.string("caloJetRef phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    dMesonRefTrackPt_HI = cms.PSet(
        pathName = cms.string(dMesonRef_HI_pathName),
        moduleName = cms.string(dMesonRef_HI_moduleName),
        plotLabel = cms.string("dMesonRefTrack_pt_HI"),
        axisLabel = cms.string("track pt [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(60)
        ),
    dMesonRefTrackEta_HI = cms.PSet(
        pathName = cms.string(dMesonRef_HI_pathName),
        moduleName = cms.string(dMesonRef_HI_moduleName),
        plotLabel = cms.string("dMesonRefTrack_eta_HI"),
        axisLabel = cms.string("track eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    dMesonRefTrackSystemMass_HI = cms.PSet(
        pathName = cms.string(dMesonRef_HI_pathName),
        moduleName = cms.string(dMesonRef_HI_moduleName),
        plotLabel = cms.string("dMesonRefTrack_systemMass_HI"),
        axisLabel = cms.string("inv mass 2 track system [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(1.34),
        Xmax = cms.double(2.6)
        ),
    dMesonRefTrackSystemPt_HI = cms.PSet(
        pathName = cms.string(dMesonRef_HI_pathName),
        moduleName = cms.string(dMesonRef_HI_moduleName),
        plotLabel = cms.string("dMesonRefTrack_systemPt_HI"),
        axisLabel = cms.string("pt 2 track system [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(5),
        Xmax = cms.double(100)
        )

)
