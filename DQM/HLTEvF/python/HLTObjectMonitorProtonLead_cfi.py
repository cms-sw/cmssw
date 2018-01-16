import FWCore.ParameterSet.Config as cms

#setup names for multiple plots that use the same paths+modules
caloJet_pathName = "HLT_PAAK4CaloJet40_Eta5p1"
caloJet_moduleName = "hltSinglePAAK4CaloJet40Eta5p1"

pfJet_pathName = "HLT_PAAK4PFJet40_Eta5p1" 
pfJet_moduleName = "hltSinglePAAK4PFJet40Eta5p1"

diCaloJet_pathName = "HLT_PADiAK4CaloJetAve40_Eta5p1" 
diCaloJet_moduleName = "hltDiAk4CaloJetAve40Eta5p1"

diPfJet_pathName = "HLT_PADiAK4PFJetAve40_Eta5p1" 
diPfJet_moduleName = "hltDiAk4PFJetAve40Eta5p1"

photon_pathName = "HLT_PASinglePhoton10_Eta3p1" 
photon_moduleName = "hltHIPhoton10Eta3p1"

photonPP_pathName = "HLT_PAPhoton10_Eta3p1_PPStyle" 
photonPP_moduleName = "hltEGBptxAND10HEFilter"

caloBJet_pathName = "HLT_PAAK4CaloBJetCSV40_Eta2p1" 
caloBJet_moduleName = "hltPABLifetimeL3FilterCSVCaloJet40Eta2p1"

pfBJet_pathName = "HLT_PAAK4PFBJetCSV40_Eta2p1" 
pfBJet_moduleName = "hltBTagPFCSVp016SingleWithMatching40"

electron_pathName = "HLT_PAEle20_WPLoose_Gsf" 
electron_moduleName = "hltPAEle20WPLooseGsfTrackIsoFilter"

l3muon3_pathName = "HLT_PAL3Mu3" 
l3muon3_moduleName = "hltL3fL1sSingleMu3BptxANDL1f0L2f0L3Filtered3"

l2muon12_pathName = "HLT_PAL2Mu12" 
l2muon12_moduleName = "hltL2fL1sSingleMu7BptxANDL1f0L2Filtered12"

l3muon12_pathName = "HLT_PAL3Mu12" 
l3muon12_moduleName = "hltL3fL1sSingleMu7BptxANDL1f0L2f0L3Filtered12"

#To avoid booking histogram, set pathName = cms.string("")

hltObjectMonitorProtonLead = DQMStep1Module('HLTObjectMonitorProtonLead',
    processName         = cms.string("HLT"),
    triggerResults      = cms.InputTag("TriggerResults", "", "HLT"),
    triggerEvent        = cms.InputTag("hltTriggerSummaryAOD", "", "HLT"),
    caloAK4JetPt = cms.PSet(
        pathName = cms.string(caloJet_pathName),
        moduleName = cms.string(caloJet_moduleName),
        plotLabel = cms.string("caloAK4Jet_pT"),
        axisLabel = cms.string("caloAK4Jet p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(0),
        Xmax = cms.double(120)
        ),
    caloAK4JetEta = cms.PSet(
        pathName = cms.string(caloJet_pathName),
        moduleName = cms.string(caloJet_moduleName),
        plotLabel = cms.string("caloAK4Jet_eta"),
        axisLabel = cms.string("caloAK4Jet eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(-5.2),
        Xmax = cms.double(5.2)
        ),
    caloAK4JetPhi = cms.PSet(
        pathName = cms.string(caloJet_pathName),
        moduleName = cms.string(caloJet_moduleName),
        plotLabel = cms.string("caloAK4Jet_phi"),
        axisLabel = cms.string("caloAK4Jet phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    pfAK4JetPt = cms.PSet(
        pathName = cms.string(pfJet_pathName),
        moduleName = cms.string(pfJet_moduleName),
        plotLabel = cms.string("pfAK4Jet_pT"),
        axisLabel = cms.string("pfAK4Jet p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(0),
        Xmax = cms.double(120)
        ),
    pfAK4JetEta = cms.PSet(
        pathName = cms.string(pfJet_pathName),
        moduleName = cms.string(pfJet_moduleName),
        plotLabel = cms.string("pfAK4Jet_eta"),
        axisLabel = cms.string("pfAK4Jet eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(-5.2),
        Xmax = cms.double(5.2)
        ),
    pfAK4JetPhi = cms.PSet(
        pathName = cms.string(pfJet_pathName),
        moduleName = cms.string(pfJet_moduleName),
        plotLabel = cms.string("pfAK4Jet_phi"),
        axisLabel = cms.string("pfAK4Jet phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    caloDiAK4JetPt = cms.PSet(
        pathName = cms.string(diCaloJet_pathName),
        moduleName = cms.string(diCaloJet_moduleName),
        plotLabel = cms.string("caloDiAK4Jet_pT"),
        axisLabel = cms.string("caloDiAK4Jet p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(0),
        Xmax = cms.double(120)
        ),
    caloDiAK4JetEta = cms.PSet(
        pathName = cms.string(diCaloJet_pathName),
        moduleName = cms.string(diCaloJet_moduleName),
        plotLabel = cms.string("caloDiAK4Jet_eta"),
        axisLabel = cms.string("caloDiAK4Jet eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(-5.2),
        Xmax = cms.double(5.2)
        ),
    caloDiAK4JetPhi = cms.PSet(
        pathName = cms.string(diCaloJet_pathName),
        moduleName = cms.string(diCaloJet_moduleName),
        plotLabel = cms.string("caloDiAK4Jet_phi"),
        axisLabel = cms.string("caloDiAK4Jet phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    pfDiAK4JetPt = cms.PSet(
        pathName = cms.string(diPfJet_pathName),
        moduleName = cms.string(diPfJet_moduleName),
        plotLabel = cms.string("pfDiAK4Jet_pT"),
        axisLabel = cms.string("pfDiAK4Jet p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(0),
        Xmax = cms.double(120)
        ),
    pfDiAK4JetEta = cms.PSet(
        pathName = cms.string(diPfJet_pathName),
        moduleName = cms.string(diPfJet_moduleName),
        plotLabel = cms.string("pfDiAK4Jet_eta"),
        axisLabel = cms.string("pfDiAK4Jet eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(75),
        Xmin = cms.double(-5.2),
        Xmax = cms.double(5.2)
        ),
    pfDiAK4JetPhi = cms.PSet(
        pathName = cms.string(diPfJet_pathName),
        moduleName = cms.string(diPfJet_moduleName),
        plotLabel = cms.string("pfDiAK4Jet_phi"),
        axisLabel = cms.string("pfDiAK4Jet phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    photonPt = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("Photon_pT"),
        axisLabel = cms.string("photon p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(0),
        Xmax = cms.double(80)
        ),
    photonEta = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("Photon_eta"),
        axisLabel = cms.string("photon eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    photonPhi = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("Photon_phi"),
        axisLabel = cms.string("photon phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    photonPPPt = cms.PSet(
        pathName = cms.string(photonPP_pathName),
        moduleName = cms.string(photonPP_moduleName),
        plotLabel = cms.string("PhotonPP_pT"),
        axisLabel = cms.string("photonPP p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(0),
        Xmax = cms.double(80)
        ),
    photonPPEta = cms.PSet(
        pathName = cms.string(photonPP_pathName),
        moduleName = cms.string(photonPP_moduleName),
        plotLabel = cms.string("PhotonPP_eta"),
        axisLabel = cms.string("photonPP eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    photonPPPhi = cms.PSet(
        pathName = cms.string(photonPP_pathName),
        moduleName = cms.string(photonPP_moduleName),
        plotLabel = cms.string("PhotonPP_phi"),
        axisLabel = cms.string("photonPP phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),

    caloBJetPt = cms.PSet(
        pathName = cms.string(caloBJet_pathName),
        moduleName = cms.string(caloBJet_moduleName),
        plotLabel = cms.string("caloBJet_pT"),
        axisLabel = cms.string("caloBJet p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(0),
        Xmax = cms.double(120)
        ),
    caloBJetEta = cms.PSet(
        pathName = cms.string(caloBJet_pathName),
        moduleName = cms.string(caloBJet_moduleName),
        plotLabel = cms.string("caloBJet_eta"),
        axisLabel = cms.string("caloBJet eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(-2.2),
        Xmax = cms.double(2.2)
        ),
    caloBJetPhi = cms.PSet(
        pathName = cms.string(caloBJet_pathName),
        moduleName = cms.string(caloBJet_moduleName),
        plotLabel = cms.string("caloBJet_phi"),
        axisLabel = cms.string("caloBJet phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    pfBJetPt = cms.PSet(
        pathName = cms.string(pfBJet_pathName),
        moduleName = cms.string(pfBJet_moduleName),
        plotLabel = cms.string("pfBJet_pT"),
        axisLabel = cms.string("pfBJet p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(60),
        Xmin = cms.double(0),
        Xmax = cms.double(120)
        ),
    pfBJetEta = cms.PSet(
        pathName = cms.string(pfBJet_pathName),
        moduleName = cms.string(pfBJet_moduleName),
        plotLabel = cms.string("pfBJet_eta"),
        axisLabel = cms.string("pfBJet eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(-2.2),
        Xmax = cms.double(2.2)
        ),
    pfBJetPhi = cms.PSet(
        pathName = cms.string(pfBJet_pathName),
        moduleName = cms.string(pfBJet_moduleName),
        plotLabel = cms.string("pfBJet_phi"),
        axisLabel = cms.string("pfBJet phi"),
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
        NbinsX = cms.int32(40),
        Xmin = cms.double(0),
        Xmax = cms.double(80)
        ),
    electronEta = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        plotLabel = cms.string("Electron_eta"),
        axisLabel = cms.string("electron eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.6),
        Xmax = cms.double(2.6)
        ),
    electronPhi = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        plotLabel = cms.string("Electron_phi"),
        axisLabel = cms.string("electron phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.4),
        Xmax = cms.double(3.4)
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
    pAL1DoubleMuZMass = cms.PSet(
	pathName = cms.string("HLT_PAL1DoubleMu10"),
	moduleName = cms.string("hltL1fL1sDoubleMu10BptxANDL1Filtered0"),
	plotLabel = cms.string("PAL1DoubleMu10_ZMass"),
	axisLabel = cms.string("L1 dimuon mass [GeV]"),
	mainWorkspace = cms.bool(True),
	NbinsX = cms.int32(50),
	Xmin = cms.double(60.0),
	Xmax = cms.double(160.0)
	),
    pAL2DoubleMuZMass = cms.PSet(
	pathName = cms.string("HLT_PAL2DoubleMu10"),
	moduleName = cms.string("hltL2fL1sDoubleMu10BptxANDL1f0L2Filtered10"),
	plotLabel = cms.string("PAL2DoubleMu10_ZMass"),
	axisLabel = cms.string("L2 dimuon mass [GeV]"),
	mainWorkspace = cms.bool(True),
	NbinsX = cms.int32(50),
	Xmin = cms.double(60.0),
	Xmax = cms.double(160.0)
	),
    pAL3DoubleMuZMass = cms.PSet(
	pathName = cms.string("HLT_PAL3DoubleMu10"),
	moduleName = cms.string("hltL3fL1sDoubleMu10BptxANDL1f0L2f10L3Filtered10"),
	plotLabel = cms.string("PAL3DoubleMu10_ZMass"),
	axisLabel = cms.string("L3 dimuon mass [GeV]"),
	mainWorkspace = cms.bool(True),
	NbinsX = cms.int32(50),
	Xmin = cms.double(60.0),
	Xmax = cms.double(160.0)
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
