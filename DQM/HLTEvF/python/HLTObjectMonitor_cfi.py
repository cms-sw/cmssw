import FWCore.ParameterSet.Config as cms

#setup names for multiple plots that use the same paths+modules
photon_pathName = "HLT_Photon30_R9Id90_HE10_IsoM"
photon_moduleName ="hltEG30R9Id90HE10IsoMTrackIsoFilter"

muon_pathName = "HLT_IsoMu27"
muon_moduleName = "hltL3crIsoL1sMu25L1f0L2f10QL3f27QL3trkIsoFiltered0p09"

electron_pathName = "HLT_Ele27_eta2p1_WP75_Gsf"
electron_moduleName = "hltEle27WP75GsfTrackIsoFilter"

caloMet_pathName = "HLT_MET75_IsoTrk50"
caloMet_moduleName = "hltMETClean75"

pfMet_pathName = "HLT_PFMET120_PFMHT120_IDTight"
pfMet_moduleName = "hltPFMET120"

jetAk8_pathName = "HLT_AK8PFJet360TrimMod_Mass30"
jetAk8_moduleName = "hltAK8SinglePFJet360TrimModMass30"

bJet_pathName = "HLT_PFMET120_NoiseCleaned_BTagCSV07"
bJet_moduleName = "hltPFMET120Filter"
bJet_pathNameOR = "HLT_QuadPFJet_SingleBTagCSV_VBF_Mqq500"
bJet_moduleNameOR = "hltCSVPF0p7"

#To avoid booking histogram, set pathName = cms.string("")

hltObjectMonitor = cms.EDAnalyzer('HLTObjectMonitor',
    alphaT = cms.PSet(
        pathName = cms.string("HLT_PFHT200_DiPFJet90_PFAlphaT0p57"),
        moduleName = cms.string("hltPFHT200PFAlphaT0p57"),
        NbinsX = cms.int32(30),
        Xmin = cms.int32(0),
        Xmax = cms.int32(5)
        ),
    photonPt = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        NbinsX = cms.int32(100),
        Xmin = cms.int32(0),
        Xmax = cms.int32(200)
        ),
    photonEta = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        NbinsX = cms.int32(50),
        Xmin = cms.int32(0),
        Xmax = cms.int32(3)
        ),
    photonPhi = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    muonPt = cms.PSet(
        pathName = cms.string(muon_pathName),
        moduleName = cms.string(muon_moduleName),
        NbinsX = cms.int32(75),
        Xmin = cms.int32(0),
        Xmax = cms.int32(150)
        ),
    muonEta = cms.PSet(
        pathName = cms.string(muon_pathName),
        moduleName = cms.string(muon_moduleName),
        NbinsX = cms.int32(50),
        Xmin = cms.int32(0),
        Xmax = cms.int32(3)
        ),
    muonPhi = cms.PSet(
        pathName = cms.string(muon_pathName),
        moduleName = cms.string(muon_moduleName),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    electronPt = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        NbinsX = cms.int32(75),
        Xmin = cms.int32(0),
        Xmax = cms.int32(150)
        ),
    electronEta = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        NbinsX = cms.int32(50),
        Xmin = cms.int32(0),
        Xmax = cms.int32(3)
        ),
    electronPhi = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    jetPt = cms.PSet(
        pathName = cms.string("HLT_PFJet200"),
        moduleName = cms.string("hltSinglePFJet200"),
        NbinsX = cms.int32(75),
        Xmin = cms.int32(150),
        Xmax = cms.int32(550)
        ),
    jetAK8Pt = cms.PSet(
        pathName = cms.string(jetAk8_pathName),
        moduleName = cms.string(jetAk8_moduleName),
        NbinsX = cms.int32(75),
        Xmin = cms.int32(150),
        Xmax = cms.int32(550)
        ),
    jetAK8Mass = cms.PSet(
        pathName = cms.string(jetAk8_pathName),
        moduleName = cms.string(jetAk8_moduleName),
        NbinsX = cms.int32(100),
        Xmin = cms.int32(0),
        Xmax = cms.int32(200)
        ),   
    tauPt = cms.PSet(
        pathName = cms.string("HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg"),
        moduleName = cms.string("hltDoublePFTau40TrackPt1MediumIsolationDz02Reg"),
        NbinsX = cms.int32(75),
        Xmin = cms.int32(30),
        Xmax = cms.int32(350)
        ),
    diMuonLowMass = cms.PSet(
        pathName = cms.string("HLT_DoubleMu4_3_Jpsi_Displaced"),
        moduleName = cms.string("hltDisplacedmumuFilterDoubleMu43Jpsi"),
        NbinsX = cms.int32(100),
        Xmin = cms.double(2.5),
        Xmax = cms.double(3.5)
        ),
    caloMetPt = cms.PSet(
        pathName = cms.string(caloMet_pathName),
        moduleName = cms.string(caloMet_moduleName),
        NbinsX = cms.int32(60),
        Xmin = cms.int32(50),
        Xmax = cms.int32(550)
        ),
    caloMetPhi = cms.PSet(
        pathName = cms.string(caloMet_pathName),
        moduleName = cms.string(caloMet_moduleName),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    pfMetPt = cms.PSet(
        pathName = cms.string(pfMet_pathName),
        moduleName = cms.string(pfMet_moduleName),
        NbinsX = cms.int32(60),
        Xmin = cms.int32(100),
        Xmax = cms.int32(500)
        ),
    pfMetPhi = cms.PSet(
        pathName = cms.string(pfMet_pathName),
        moduleName = cms.string(pfMet_moduleName),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    caloHtPt = cms.PSet(
        pathName = cms.string("HLT_HT650_DisplacedDijet80_Inclusive"),
        moduleName = cms.string("hltHT650"),
        NbinsX = cms.int32(200),
        Xmin = cms.int32(0),
        Xmax = cms.int32(2000)
        ),
    pfHtPt = cms.PSet(
        pathName = cms.string("HLT_PFHT750_4Jet"),
        moduleName = cms.string("hltPF4JetHT750"),
        NbinsX = cms.int32(200),
        Xmin = cms.int32(0),
        Xmax = cms.int32(2000)
        ),
    bJetEta = cms.PSet(
        pathName = cms.string(bJet_pathName),
        moduleName = cms.string(bJet_moduleName),
        pathName_OR = cms.string(bJet_pathNameOR),
        moduleName_OR = cms.string(bJet_moduleNameOR),
        NbinsX = cms.int32(50),
        Xmin = cms.int32(0),
        Xmax = cms.int32(3)
        ),
    bJetPhi = cms.PSet(
        pathName = cms.string(bJet_pathName),
        moduleName = cms.string(bJet_moduleName),
        pathName_OR = cms.string(bJet_pathNameOR),
        moduleName_OR = cms.string(bJet_moduleNameOR),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    diMuonMass = cms.PSet(
        pathName = cms.string("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ"),
        moduleName = cms.string("hltDiMuonGlb17Glb8RelTrkIsoFiltered0p4DzFiltered0p2"),
        pathName_OR = cms.string("HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ"),
        moduleName_OR = cms.string("hltDiMuonGlb17Trk8RelTrkIsoFiltered0p4DzFiltered0p2"),
        NbinsX = cms.int32(50),
        Xmin = cms.int32(60),
        Xmax = cms.int32(160)
        ),
    diElecMass = cms.PSet(
        pathName = cms.string("HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"),
        moduleName = cms.string("hltEle23Ele12CaloIdLTrackIdLIsoVLDZFilter"),
        NbinsX = cms.int32(50),
        Xmin = cms.int32(60),
        Xmax = cms.int32(160)
        )
)
