import FWCore.ParameterSet.Config as cms

#setup names for multiple plots that use the same paths+modules
pfJet_pathName = "HLT_AK4PFJet40"
pfJet_moduleName = "hltSingleAK4PFJet40"

caloJet_pathName = "HLT_AK4CaloJet40"
caloJet_moduleName = "hltSingleAK4CaloJet40"

pfJetFWD_pathName = "HLT_AK4PFJet40FWD"
pfJetFWD_moduleName = "hltSingleAK4PFJet40FWD"

caloJetFWD_pathName = "HLT_AK4CaloJet40FWD"
caloJetFWD_moduleName = "hltSingleAK4CaloJet40FWD"

pfBJet_pathName = "HLT_AK4PFJet40_bTag"
pfBJet_moduleName = "hltBTagPFCSV0p80SingleJet40Eta2p4"

photon_pathName = "HLT_HISinglePhoton10_Eta3p1ForPPRef"
photon_moduleName = "hltHIPhoton10Eta3p1"

photonHELoose_pathName = "HLT_Photon20_HoverELoose"
photonHELoose_moduleName = "hltEG20HEFilterLooseHoverE"

electron_pathName = "HLT_Ele15_WPLoose_Gsf"
electron_moduleName = "hltEle15WPLoose1GsfTrackIsoFilter"

l3muon3_pathName = "HLT_HIL3Mu3"
l3muon3_moduleName = "hltL3fL1sSingleMu3L1f0L2f0L3Filtered3"

l2muon12_pathName = "HLT_HIL2Mu12"
l2muon12_moduleName = "hltL2fL1sSingleMu7L1f0L2Filtered12"

l3muon12_pathName = "HLT_HIL3Mu12"
l3muon12_moduleName = "hltL3fL1sSingleMu7L1f0L2f0L3Filtered12"

l1doublemuon10_pathName = "HLT_HIL1DoubleMu10"
l1doublemuon10_moduleName = "hltL1fL1sDoubleMu10L1Filtered0"

l2doublemuon10_pathName = "HLT_HIL2DoubleMu10"
l2doublemuon10_moduleName = "hltL2fL1sDoubleMu10L1f0L2Filtered10"

l3doublemuon10_pathName = "HLT_HIL3DoubleMu10"
l3doublemuon10_moduleName = "hltL3fL1sDoubleMu10L1f0L2f0L3Filtered10"

#To avoid booking histogram, set pathName = cms.string("")

hltObjectMonitorPPReference = cms.EDAnalyzer('HLTObjectMonitorPPReference',
    processName         = cms.string("HLT"),
    triggerResults      = cms.InputTag("TriggerResults", "", "HLT"),
    triggerEvent        = cms.InputTag("hltTriggerSummaryAOD", "", "HLT"),
    caloAK4JetPt = cms.PSet(
        pathName = cms.string(caloJet_pathName),
        moduleName = cms.string(caloJet_moduleName),
        plotLabel = cms.string("caloAK4Jet40_pT"),
        axisLabel = cms.string("caloAK4Jet40 p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(0),
        Xmax = cms.double(120)
        ),
    caloAK4JetEta = cms.PSet(
        pathName = cms.string(caloJet_pathName),
        moduleName = cms.string(caloJet_moduleName),
        plotLabel = cms.string("caloAK4Jet40_eta"),
        axisLabel = cms.string("caloAK4Jet40 eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(-5.2),
        Xmax = cms.double(5.2)
        ),
    caloAK4JetPhi = cms.PSet(
        pathName = cms.string(caloJet_pathName),
        moduleName = cms.string(caloJet_moduleName),
        plotLabel = cms.string("caloAK4Jet40_phi"),
        axisLabel = cms.string("caloAK4Jet40 phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    pfAK4JetPt = cms.PSet(
        pathName = cms.string(pfJet_pathName),
        moduleName = cms.string(pfJet_moduleName),
        plotLabel = cms.string("pfAK4Jet40_pT"),
        axisLabel = cms.string("pfAK4Jet40 p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(0),
        Xmax = cms.double(120)
        ),
    pfAK4JetEta = cms.PSet(
        pathName = cms.string(pfJet_pathName),
        moduleName = cms.string(pfJet_moduleName),
        plotLabel = cms.string("pfAK4Jet40_eta"),
        axisLabel = cms.string("pfAK4Jet40 eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(-5.2),
        Xmax = cms.double(5.2)
        ),
    pfAK4JetPhi = cms.PSet(
        pathName = cms.string(pfJet_pathName),
        moduleName = cms.string(pfJet_moduleName),
        plotLabel = cms.string("pfAK4Jet40_phi"),
        axisLabel = cms.string("pfAK4Jet40 phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    caloAK4JetFWDPt = cms.PSet(
        pathName = cms.string(caloJetFWD_pathName),
        moduleName = cms.string(caloJetFWD_moduleName),
        plotLabel = cms.string("caloAK4Jet40FWD_pT"),
        axisLabel = cms.string("caloAK4Jet40FWD p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(0),
        Xmax = cms.double(120)
        ),
    caloAK4JetFWDEta = cms.PSet(
        pathName = cms.string(caloJetFWD_pathName),
        moduleName = cms.string(caloJetFWD_moduleName),
        plotLabel = cms.string("caloAK4Jet40FWD_eta"),
        axisLabel = cms.string("caloAK4Jet40FWD eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(-5.2),
        Xmax = cms.double(5.2)
        ),
    caloAK4JetFWDPhi = cms.PSet(
        pathName = cms.string(caloJetFWD_pathName),
        moduleName = cms.string(caloJetFWD_moduleName),
        plotLabel = cms.string("caloAK4Jet40FWD_phi"),
        axisLabel = cms.string("caloAK4Jet40FWD phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    pfAK4JetFWDPt = cms.PSet(
        pathName = cms.string(pfJetFWD_pathName),
        moduleName = cms.string(pfJetFWD_moduleName),
        plotLabel = cms.string("pfAK4Jet40FWD_pT"),
        axisLabel = cms.string("pfAK4Jet40FWD p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(0),
        Xmax = cms.double(120)
        ),
    pfAK4JetFWDEta = cms.PSet(
        pathName = cms.string(pfJetFWD_pathName),
        moduleName = cms.string(pfJetFWD_moduleName),
        plotLabel = cms.string("pfAK4Jet40FWD_eta"),
        axisLabel = cms.string("pfAK4Jet40FWD eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(-5.2),
        Xmax = cms.double(5.2)
        ),
    pfAK4JetFWDPhi = cms.PSet(
        pathName = cms.string(pfJetFWD_pathName),
        moduleName = cms.string(pfJetFWD_moduleName),
        plotLabel = cms.string("pfAK4Jet40FWD_phi"),
        axisLabel = cms.string("pfAK4Jet40FWD phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    pfBJetPt = cms.PSet(
        pathName = cms.string(pfBJet_pathName),
        moduleName = cms.string(pfBJet_moduleName),
        plotLabel = cms.string("pfAK4Jet40Bjet_pT"),
        axisLabel = cms.string("pfAK4Jet40Bjet p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(0),
        Xmax = cms.double(120)
        ),
    pfBJetEta = cms.PSet(
        pathName = cms.string(pfBJet_pathName),
        moduleName = cms.string(pfBJet_moduleName),
        plotLabel = cms.string("pfAK4Jet40Bjet_eta"),
        axisLabel = cms.string("pfAK4Jet40Bjet eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(-2.2),
        Xmax = cms.double(2.2)
        ),
    pfBJetPhi = cms.PSet(
        pathName = cms.string(pfBJet_pathName),
        moduleName = cms.string(pfBJet_moduleName),
        plotLabel = cms.string("pfAK4Jet40Bjet_phi"),
        axisLabel = cms.string("pfAK4Jet40Bjet phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    photonHELoosePt = cms.PSet(
        pathName = cms.string(photonHELoose_pathName),
        moduleName = cms.string(photonHELoose_moduleName),
        plotLabel = cms.string("Photon20HoverELoose_pT"),
        axisLabel = cms.string("photon20HoverELoose p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(25),
        Xmin = cms.double(0),
        Xmax = cms.double(50)
        ),
    photonHELooseEta = cms.PSet(
        pathName = cms.string(photonHELoose_pathName),
        moduleName = cms.string(photonHELoose_moduleName),
        plotLabel = cms.string("Photon20HoverELoose_eta"),
        axisLabel = cms.string("Photon20HoverELoose eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    photonHELoosePhi = cms.PSet(
        pathName = cms.string(photonHELoose_pathName),
        moduleName = cms.string(photonHELoose_moduleName),
        plotLabel = cms.string("Photon20HoverELoose_phi"),
        axisLabel = cms.string("Photon20HoverELoose phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    photonPt = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("Photon10_pT"),
        axisLabel = cms.string("photon10 p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(25),
        Xmin = cms.double(0),
        Xmax = cms.double(50)
        ),
    photonEta = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("Photon10_eta"),
        axisLabel = cms.string("photon10 eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    photonPhi = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("Photon10_phi"),
        axisLabel = cms.string("photon10 phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    electronPt = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        plotLabel = cms.string("Electron_pT"),
        axisLabel = cms.string("electron p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(25),
        Xmin = cms.double(0),
        Xmax = cms.double(50)
        ),
    electronEta = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        plotLabel = cms.string("Electron_eta"),
        axisLabel = cms.string("electron eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    electronPhi = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        plotLabel = cms.string("Electron_phi"),
        axisLabel = cms.string("electron phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    l3muon3Pt = cms.PSet(
        pathName = cms.string(l3muon3_pathName),
        moduleName = cms.string(l3muon3_moduleName),
        plotLabel = cms.string("l3muon3_pT"),
        axisLabel = cms.string("l3muon3 p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(25),
        Xmin = cms.double(0),
        Xmax = cms.double(50)
        ),
    l3muon3Eta = cms.PSet(
        pathName = cms.string(l3muon3_pathName),
        moduleName = cms.string(l3muon3_moduleName),
        plotLabel = cms.string("l3muon3_eta"),
        axisLabel = cms.string("l3muon3 eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    l3muon3Phi = cms.PSet(
        pathName = cms.string(l3muon3_pathName),
        moduleName = cms.string(l3muon3_moduleName),
        plotLabel = cms.string("l3muon3_phi"),
        axisLabel = cms.string("l3muon3 phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    l2muon12Pt = cms.PSet(
        pathName = cms.string(l2muon12_pathName),
        moduleName = cms.string(l2muon12_moduleName),
        plotLabel = cms.string("l2muon12_pT"),
        axisLabel = cms.string("l2muon12 p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(25),
        Xmin = cms.double(0),
        Xmax = cms.double(50)
        ),
    l2muon12Eta = cms.PSet(
        pathName = cms.string(l2muon12_pathName),
        moduleName = cms.string(l2muon12_moduleName),
        plotLabel = cms.string("l2muon12_eta"),
        axisLabel = cms.string("l2muon12 eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    l2muon12Phi = cms.PSet(
        pathName = cms.string(l2muon12_pathName),
        moduleName = cms.string(l2muon12_moduleName),
        plotLabel = cms.string("l2muon12_phi"),
        axisLabel = cms.string("l2muon12 phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    l3muon12Pt = cms.PSet(
        pathName = cms.string(l3muon12_pathName),
        moduleName = cms.string(l3muon12_moduleName),
        plotLabel = cms.string("l3muon12_pT"),
        axisLabel = cms.string("l3muon12 p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(25),
        Xmin = cms.double(0),
        Xmax = cms.double(50)
        ),
    l3muon12Eta = cms.PSet(
        pathName = cms.string(l3muon12_pathName),
        moduleName = cms.string(l3muon12_moduleName),
        plotLabel = cms.string("l3muon12_eta"),
        axisLabel = cms.string("l3muon12 eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    l3muon12Phi = cms.PSet(
        pathName = cms.string(l3muon12_pathName),
        moduleName = cms.string(l3muon12_moduleName),
        plotLabel = cms.string("l3muon12_phi"),
        axisLabel = cms.string("l3muon12 phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),

    l1doublemuon10Mass = cms.PSet(
        pathName = cms.string(l1doublemuon10_pathName),
        moduleName = cms.string(l1doublemuon10_moduleName),
        plotLabel = cms.string("l1doublemuon10_mass"),
        axisLabel = cms.string("l1doublemuon10 mass [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(60),
        Xmax = cms.double(160)
        ),
    l2doublemuon10Mass = cms.PSet(
        pathName = cms.string(l2doublemuon10_pathName),
        moduleName = cms.string(l2doublemuon10_moduleName),
        plotLabel = cms.string("l2doublemuon10_mass"),
        axisLabel = cms.string("l2doublemuon10 mass [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(60),
        Xmax = cms.double(160)
        ),
    l3doublemuon10Mass = cms.PSet(
        pathName = cms.string(l3doublemuon10_pathName),
        moduleName = cms.string(l3doublemuon10_moduleName),
        plotLabel = cms.string("l3doublemuon10_mass"),
        axisLabel = cms.string("l3doublemuon10 mass [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(60),
        Xmax = cms.double(160)
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
