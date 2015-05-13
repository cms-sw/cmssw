import FWCore.ParameterSet.Config as cms

hltWorkspace = cms.EDAnalyzer('HLTWorkspace',
    alphaT = cms.PSet(
        pathName = cms.string("HLT_PFHT200_DiPFJet90_PFAlphaT0p57"),
        moduleName = cms.string("hltPFHT200PFAlphaT0p57"),
        NbinsX = cms.int32(30),
        Xmin = cms.int32(0),
        Xmax = cms.int32(5)
        ),
    photonPt = cms.PSet(
        pathName = cms.string("HLT_Photon30_R9Id90_HE10_IsoM"),
        moduleName = cms.string("hltEG30R9Id90HE10IsoMTrackIsoFilter"),
        NbinsX = cms.int32(100),
        Xmin = cms.int32(0),
        Xmax = cms.int32(200)
        ),
    photonEta = cms.PSet(
        pathName = cms.string("HLT_Photon30_R9Id90_HE10_IsoM"),
        moduleName = cms.string("hltEG30R9Id90HE10IsoMTrackIsoFilter"),
        NbinsX = cms.int32(50),
        Xmin = cms.int32(0),
        Xmax = cms.int32(3)
        ),
    photonPhi = cms.PSet(
        pathName = cms.string("HLT_Photon30_R9Id90_HE10_IsoM"),
        moduleName = cms.string("hltEG30R9Id90HE10IsoMTrackIsoFilter"),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    muonPt = cms.PSet(
        pathName = cms.string("HLT_IsoMu27"),
        moduleName = cms.string("hltL3crIsoL1sMu25L1f0L2f10QL3f27QL3trkIsoFiltered0p09"),
        NbinsX = cms.int32(75),
        Xmin = cms.int32(0),
        Xmax = cms.int32(150)
        ),
    muonEta = cms.PSet(
        pathName = cms.string("HLT_IsoMu27"),
        moduleName = cms.string("hltL3crIsoL1sMu25L1f0L2f10QL3f27QL3trkIsoFiltered0p09"),
        NbinsX = cms.int32(50),
        Xmin = cms.int32(0),
        Xmax = cms.int32(3)
        ),
    muonPhi = cms.PSet(
        pathName = cms.string("HLT_IsoMu27"),
        moduleName = cms.string("hltL3crIsoL1sMu25L1f0L2f10QL3f27QL3trkIsoFiltered0p09"),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    electronPt = cms.PSet(
        pathName = cms.string("HLT_Ele27_eta2p1_WP75_Gsf"),
        moduleName = cms.string("hltEle27WP75GsfTrackIsoFilter"),
        NbinsX = cms.int32(75),
        Xmin = cms.int32(0),
        Xmax = cms.int32(150)
        ),
    electronEta = cms.PSet(
        pathName = cms.string("HLT_Ele27_eta2p1_WP75_Gsf"),
        moduleName = cms.string("hltEle27WP75GsfTrackIsoFilter"),
        NbinsX = cms.int32(50),
        Xmin = cms.int32(0),
        Xmax = cms.int32(3)
        ),
    electronPhi = cms.PSet(
        pathName = cms.string("HLT_Ele27_eta2p1_WP75_Gsf"),
        moduleName = cms.string("hltEle27WP75GsfTrackIsoFilter"),
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
        pathName = cms.string("HLT_MET75_IsoTrk50"),
        moduleName = cms.string("hltMETClean75"),
        NbinsX = cms.int32(60),
        Xmin = cms.int32(50),
        Xmax = cms.int32(550)
        ),
    caloMetPhi = cms.PSet(
        pathName = cms.string("HLT_MET75_IsoTrk50"),
        moduleName = cms.string("hltMETClean75"),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        ),
    pfMetPt = cms.PSet(
        pathName = cms.string("HLT_PFMET120_PFMHT120_IDTight"),
        moduleName = cms.string("hltPFMET120"),
        NbinsX = cms.int32(60),
        Xmin = cms.int32(100),
        Xmax = cms.int32(500)
        ),
    pfMetPhi = cms.PSet(
        pathName = cms.string("HLT_PFMET120_PFMHT120_IDTight"),
        moduleName = cms.string("hltPFMET120"),
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
        pathName = cms.string("HLT_PFMET120_NoiseCleaned_BTagCSV07"),
        moduleName = cms.string("hltPFMET120Filter"),
        pathName_OR = cms.string("HLT_QuadPFJet_SingleBTagCSV_VBF_Mqq500"),
        moduleName_OR = cms.string("hltCSVPF0p7"),
        NbinsX = cms.int32(50),
        Xmin = cms.int32(0),
        Xmax = cms.int32(3)
        ),
    bJetPhi = cms.PSet(
        pathName = cms.string("HLT_PFMET120_NoiseCleaned_BTagCSV07"),
        moduleName = cms.string("hltPFMET120Filter"),
        pathName_OR = cms.string("HLT_QuadPFJet_SingleBTagCSV_VBF_Mqq500"),
        moduleName_OR = cms.string("hltCSVPF0p7"),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
        )
)
