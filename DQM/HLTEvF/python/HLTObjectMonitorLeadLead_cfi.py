import FWCore.ParameterSet.Config as cms

#setup names for multiple plots that use the same paths+modules

caloJet_pathName = "HLT_HIPuAK4CaloJet40Eta5p1" 
caloJet_moduleName = "hltSinglePuAK4CaloJet40Eta5p1"

cspfJet_pathName = "HLT_HICsAK4PFJet60Eta1p5"
cspfJet_moduleName = "hltSingleCsPFJet60Eta1p5"

caloBJetdeepcsv_pathName = "HLT_HIPuAK4CaloJet60Eta2p4_DeepCSV0p4" 
caloBJetdeepcsv_moduleName = "hltBTagCaloDeepCSV0p4TagSingleJet60"

caloBJetcsv_pathName = "HLT_HIPuAK4CaloJet60Eta2p4_CSVv2WP0p75" 
caloBJetcsv_moduleName = "hltBTagCaloCSVv2WP0p75SingleJet60HI"

photon_pathName = "HLT_HIGEDPhoton10"
photon_moduleName = "hltEG10HoverELoosePPOnAAFilter"

isphoton_pathName = "HLT_HIIslandPhoton10_Eta3p1"
isphoton_moduleName = "hltHIIslandPhoton10Eta3p1"

electron_pathName = "HLT_HIEle15Gsf"
electron_moduleName = "hltEle15GsfTrackIsoPPOnAAFilter"

zbsinglepixel_pathName = "HLT_HIUPC_ZeroBias_SinglePixelTrack"
zbsinglepixel_moduleName = "hltPixelTracksForUPCFilterPPOnAA"

hfvetomaxtrack_pathName = "HLT_HIUPC_NotMBHF2OR_BptxAND_SinglePixelTrack_MaxPixelTrack"
hfvetomaxtrack_moduleName = "hltPixelTracksForUPCFilterPPOnAA"

hfvetosgltrack_pathName = "HLT_HIUPC_NotMBHF2OR_BptxAND_SinglePixelTrack"
hfvetosgltrack_moduleName = "hltPixelTracksForUPCFilterPPOnAA"

sgleg5_pathName = "HLT_HIUPC_SingleEG5_NotMBHF2AND_SinglePixelTrack_MaxPixelTrack"
sgleg5_moduleName = "hltPixelTracksForUPCFilterPPOnAA"

muopenhfveto_pathName = "HLT_HIUPC_SingleMuOpen_NotMBHF2OR_MaxPixelTrack"
muopenhfveto_moduleName = "hltPixelTracksForUPCFilterPPOnAA"

mu0hfveto_pathName = "HLT_HIUPC_SingleMu0_NotMBHF2AND_MaxPixelTrack"
mu0hfveto_moduleName = "hltPixelTracksForUPCFilterPPOnAA"

doublemuhfveto_pathName = "HLT_HIUPC_DoubleMu0_NotMBHF2AND_MaxPixelTrack"
doublemuhfveto_moduleName = "hltPixelTracksForUPCFilterPPOnAA"

tracking1_pathName = "HLT_HIDmesonPPTrackingGlobal_Dpt20"
tracking1_moduleName = "hlttktkFilterForDmesonDpt20"
tracking2_pathName = "HLT_HIDsPPTrackingGlobal_Dpt20"
tracking2_moduleName = "hlttktkFilterForDsDpt20"
tracking3_pathName = "HLT_HILcPPTrackingGlobal_Dpt20"
tracking3_moduleName = "hlttktkFilterForLcDpt20"

fulltrack_pathName = "HLT_HIFullTracks_Multiplicity6080"
fulltrack_moduleName = "hltFullIterativeTrackingMergedPPOnAA"

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hltObjectMonitorLeadLead = DQMEDAnalyzer('HLTObjectMonitorHI',
    processName         = cms.string("HLT"),
    triggerResults      = cms.InputTag("TriggerResults", "", "HLT"),
    triggerEvent        = cms.InputTag("hltTriggerSummaryAOD", "", "HLT"),

    HIPUCaloJet40Pt = cms.PSet(
        pathName = cms.string(caloJet_pathName),
        moduleName = cms.string(caloJet_moduleName),
        plotLabel = cms.string("HIPUCaloJet40_pT"),
        axisLabel = cms.string("HIPUCaloJet40 p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(100),
        Xmin = cms.double(0),
        Xmax = cms.double(200)
        ),
    HIPUCaloJet40Eta = cms.PSet(
        pathName = cms.string(caloJet_pathName),
        moduleName = cms.string(caloJet_moduleName),
        plotLabel = cms.string("HIPUCaloJet40_eta"),
        axisLabel = cms.string("HIPUCaloJet40 eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(32),
        Xmin = cms.double(-5.1),
        Xmax = cms.double(5.1)
        ),
    HIPUCaloJet40Phi = cms.PSet(
        pathName = cms.string(caloJet_pathName),
        moduleName = cms.string(caloJet_moduleName),
        plotLabel = cms.string("HIPUCaloJet40_phi"),
        axisLabel = cms.string("HIPUCaloJet40 phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HICSPFJet60Pt = cms.PSet(
        pathName = cms.string(cspfJet_pathName),
        moduleName = cms.string(cspfJet_moduleName),
        plotLabel = cms.string("HICSPFJet60_pT"),
        axisLabel = cms.string("HICSPFJet60 p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(100),
        Xmin = cms.double(0),
        Xmax = cms.double(200)
        ),
    HICSPFJet60Eta = cms.PSet(
        pathName = cms.string(cspfJet_pathName),
        moduleName = cms.string(cspfJet_moduleName),
        plotLabel = cms.string("HICSPFJet60_eta"),
        axisLabel = cms.string("HICSPFJet60 eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(32),
        Xmin = cms.double(-1.5),
        Xmax = cms.double(1.5)
        ),
    HICSPFJet60Phi = cms.PSet(
        pathName = cms.string(cspfJet_pathName),
        moduleName = cms.string(cspfJet_moduleName),
        plotLabel = cms.string("HICSPFJet60_phi"),
        axisLabel = cms.string("HICSPFJet60 phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HIPUCaloBJet60DeepCSVPt = cms.PSet(
        pathName = cms.string(caloBJetdeepcsv_pathName),
        moduleName = cms.string(caloBJetdeepcsv_moduleName),
        plotLabel = cms.string("HIPUCaloBJet60DeepCSV_pT"),
        axisLabel = cms.string("HIPUCaloBJet60DeepCSV p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(100),
        Xmin = cms.double(0),
        Xmax = cms.double(200)
        ),
    HIPUCaloBJet60DeepCSVEta = cms.PSet(
        pathName = cms.string(caloBJetdeepcsv_pathName),
        moduleName = cms.string(caloBJetdeepcsv_moduleName),
        plotLabel = cms.string("HIPUCaloBJet60DeepCSV_eta"),
        axisLabel = cms.string("HIPUCaloBJet60DeepCSV eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(32),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    HIPUCaloBJet60DeepCSVPhi = cms.PSet(
        pathName = cms.string(caloBJetdeepcsv_pathName),
        moduleName = cms.string(caloBJetdeepcsv_moduleName),
        plotLabel = cms.string("HIPUCaloBJet60DeepCSV_phi"),
        axisLabel = cms.string("HIPUCaloBJet60DeepCSV phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HIPUCaloBJet60CSVv2Pt = cms.PSet(
        pathName = cms.string(caloBJetcsv_pathName),
        moduleName = cms.string(caloBJetcsv_moduleName),
        plotLabel = cms.string("HIPUCaloBJet60CSVv2_pT"),
        axisLabel = cms.string("HIPUCaloBJet60CSVv2 p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(100),
        Xmin = cms.double(0),
        Xmax = cms.double(200)
        ),
    HIPUCaloBJet60CSVv2Eta = cms.PSet(
        pathName = cms.string(caloBJetcsv_pathName),
        moduleName = cms.string(caloBJetcsv_moduleName),
        plotLabel = cms.string("HIPUCaloBJet60CSVv2_eta"),
        axisLabel = cms.string("HIPUCaloBJet60CSVv2 eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(32),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    HIPUCaloBJet60CSVv2Phi = cms.PSet(
        pathName = cms.string(caloBJetcsv_pathName),
        moduleName = cms.string(caloBJetcsv_moduleName),
        plotLabel = cms.string("HIPUCaloBJet60CSVv2_phi"),
        axisLabel = cms.string("HIPUCaloBJet60CSVv2 phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    photonPt = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("HIGEDPhoton10_pT"),
        axisLabel = cms.string("GED photon p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(0),
        Xmax = cms.double(80)
        ),
    photonEta = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("HIGEDPhoton10_eta"),
        axisLabel = cms.string("GED photon eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(32),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    photonPhi = cms.PSet(
        pathName = cms.string(photon_pathName),
        moduleName = cms.string(photon_moduleName),
        plotLabel = cms.string("HIGEDPhoton10_phi"),
        axisLabel = cms.string("GED photon phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    isphotonPt = cms.PSet(
        pathName = cms.string(isphoton_pathName),
        moduleName = cms.string(isphoton_moduleName),
        plotLabel = cms.string("HIIslandPhoton10_pT"),
        axisLabel = cms.string("Island photon p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(0),
        Xmax = cms.double(80)
        ),
    isphotonEta = cms.PSet(
        pathName = cms.string(isphoton_pathName),
        moduleName = cms.string(isphoton_moduleName),
        plotLabel = cms.string("HIIslandPhoton10_eta"),
        axisLabel = cms.string("Island photon eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(32),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    isphotonPhi = cms.PSet(
        pathName = cms.string(isphoton_pathName),
        moduleName = cms.string(isphoton_moduleName),
        plotLabel = cms.string("HIIslandPhoton10_phi"),
        axisLabel = cms.string("Island photon phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    electronPt = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        plotLabel = cms.string("HIEle15Gsf_pT"),
        axisLabel = cms.string("electron p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(0),
        Xmax = cms.double(80)
        ),
    electronEta = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        plotLabel = cms.string("HIEle15Gsf_eta"),
        axisLabel = cms.string("electron eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    electronPhi = cms.PSet(
        pathName = cms.string(electron_pathName),
        moduleName = cms.string(electron_moduleName),
        plotLabel = cms.string("HIEle15Gsf_phi"),
        axisLabel = cms.string("electron phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HIZBSinglePixelPt = cms.PSet(
        pathName = cms.string(zbsinglepixel_pathName),
        moduleName = cms.string(zbsinglepixel_moduleName),
        plotLabel = cms.string("HIZBSinglePixel_pT"),
        axisLabel = cms.string("p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(20)
        ),
    HIZBSinglePixelEta = cms.PSet(
        pathName = cms.string(zbsinglepixel_pathName),
        moduleName = cms.string(zbsinglepixel_moduleName),
        plotLabel = cms.string("HIZBSinglePixel_eta"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    HIZBSinglePixelPhi = cms.PSet(
        pathName = cms.string(zbsinglepixel_pathName),
        moduleName = cms.string(zbsinglepixel_moduleName),
        plotLabel = cms.string("HIZBSinglePixel_phi"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HIHFVetoInORSinglePixelMaxTrackPt = cms.PSet(
        pathName = cms.string(hfvetomaxtrack_pathName),
        moduleName = cms.string(hfvetomaxtrack_moduleName),
        plotLabel = cms.string("HIHFVetoInORSinglePixelMaxTrack_pT"),
        axisLabel = cms.string("p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(20)
        ),
    HIHFVetoInORSinglePixelMaxTrackEta = cms.PSet(
        pathName = cms.string(hfvetomaxtrack_pathName),
        moduleName = cms.string(hfvetomaxtrack_moduleName),
        plotLabel = cms.string("HIHFVetoInORSinglePixelMaxTrack_eta"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    HIHFVetoInORSinglePixelMaxTrackPhi = cms.PSet(
        pathName = cms.string(hfvetomaxtrack_pathName),
        moduleName = cms.string(hfvetomaxtrack_moduleName),
        plotLabel = cms.string("HIHFVetoInORSinglePixelMaxTrack_phi"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HIHFVetoInORSinglePixelPt = cms.PSet(
        pathName = cms.string(hfvetosgltrack_pathName),
        moduleName = cms.string(hfvetosgltrack_moduleName),
        plotLabel = cms.string("HIHFVetoInORSinglePixel_pT"),
        axisLabel = cms.string("p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(20)
        ),
    HIHFVetoInORSinglePixelEta = cms.PSet(
        pathName = cms.string(hfvetosgltrack_pathName),
        moduleName = cms.string(hfvetosgltrack_moduleName),
        plotLabel = cms.string("HIHFVetoInORSinglePixel_eta"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    HIHFVetoInORSinglePixelPhi = cms.PSet(
        pathName = cms.string(hfvetosgltrack_pathName),
        moduleName = cms.string(hfvetosgltrack_moduleName),
        plotLabel = cms.string("HIHFVetoInORSinglePixel_phi"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HISingleEG5HFVetoInAndSinglePixelMaxTrackPt = cms.PSet(
        pathName = cms.string(sgleg5_pathName),
        moduleName = cms.string(sgleg5_moduleName),
        plotLabel = cms.string("HISingleEG5HFVetoInAndSinglePixelMaxTrack_pT"),
        axisLabel = cms.string("p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(20)
        ),
    HISingleEG5HFVetoInAndSinglePixelMaxTrackEta = cms.PSet(
        pathName = cms.string(sgleg5_pathName),
        moduleName = cms.string(sgleg5_moduleName),
        plotLabel = cms.string("HISingleEG5HFVetoInAndSinglePixelMaxTrack_eta"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    HISingleEG5HFVetoInAndSinglePixelMaxTrackPhi = cms.PSet(
        pathName = cms.string(sgleg5_pathName),
        moduleName = cms.string(sgleg5_moduleName),
        plotLabel = cms.string("HISingleEG5HFVetoInAndSinglePixelMaxTrack_phi"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HIMuOpenHFVetoInORMaxTrackPt = cms.PSet(
        pathName = cms.string(muopenhfveto_pathName),
        moduleName = cms.string(muopenhfveto_moduleName),
        plotLabel = cms.string("HIMuOpenHFVetoInORMaxTrack_pT"),
        axisLabel = cms.string("p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(20)
        ),
    HIMuOpenHFVetoInORMaxTrackEta = cms.PSet(
        pathName = cms.string(muopenhfveto_pathName),
        moduleName = cms.string(muopenhfveto_moduleName),
        plotLabel = cms.string("HIMuOpenHFVetoInORMaxTrack_eta"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    HIMuOpenHFVetoInORMaxTrackPhi = cms.PSet(
        pathName = cms.string(muopenhfveto_pathName),
        moduleName = cms.string(muopenhfveto_moduleName),
        plotLabel = cms.string("HIMuOpenHFVetoInORMaxTrack_phi"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HIMu0HFVetoInAndMaxTrackPt = cms.PSet(
        pathName = cms.string(mu0hfveto_pathName),
        moduleName = cms.string(mu0hfveto_moduleName),
        plotLabel = cms.string("HIMu0HFVetoInAndMaxTrack_pT"),
        axisLabel = cms.string("p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(20)
        ),
    HIMu0HFVetoInAndMaxTrackEta = cms.PSet(
        pathName = cms.string(mu0hfveto_pathName),
        moduleName = cms.string(mu0hfveto_moduleName),
        plotLabel = cms.string("HIMu0HFVetoInAndMaxTrack_eta"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    HIMu0HFVetoInAndMaxTrackPhi = cms.PSet(
        pathName = cms.string(mu0hfveto_pathName),
        moduleName = cms.string(mu0hfveto_moduleName),
        plotLabel = cms.string("HIMu0HFVetoInAndMaxTrack_phi"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HIDoubleMu0HFVetoInAndMaxTrackPt = cms.PSet(
        pathName = cms.string(doublemuhfveto_pathName),
        moduleName = cms.string(doublemuhfveto_moduleName),
        plotLabel = cms.string("HIDoubleMu0HFVetoInAndMaxTrack_pT"),
        axisLabel = cms.string("dimu p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(40),
        Xmin = cms.double(0),
        Xmax = cms.double(2)
        ),
    HIDoubleMu0HFVetoInAndMaxTrackEta = cms.PSet(
        pathName = cms.string(doublemuhfveto_pathName),
        moduleName = cms.string(doublemuhfveto_moduleName),
        plotLabel = cms.string("HIDoubleMu0HFVetoInAndMaxTrack_eta"),
        axisLabel = cms.string("dimu eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    HIDoubleMu0HFVetoInAndMaxTrackMass = cms.PSet(
        pathName = cms.string(doublemuhfveto_pathName),
        moduleName = cms.string(doublemuhfveto_moduleName),
        plotLabel = cms.string("HIDoubleMu0HFVetoInAndMaxTrack_Mass"),
        axisLabel = cms.string("dimu mass"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(25),
        Xmin = cms.double(2),
        Xmax = cms.double(5)
        ),
    HIL1DoubleMuZMass = cms.PSet(
	pathName = cms.string("HLT_HIL1DoubleMu10"),
	moduleName = cms.string("hltL1fL1sL1DoubleMu10L1Filtered0"),
	plotLabel = cms.string("HIL1DoubleMu10_ZMass"),
	axisLabel = cms.string("L1 dimuon mass [GeV]"),
	mainWorkspace = cms.bool(True),
	NbinsX = cms.int32(50),
	Xmin = cms.double(60.0),
	Xmax = cms.double(160.0)
	),
    HIL2DoubleMuZMass = cms.PSet(
	pathName = cms.string("HLT_HIL2_L1DoubleMu10"),
	moduleName = cms.string("hltL2fL1sL1DoubleMu10L1f0L2Filtered0"),
	plotLabel = cms.string("HIL2DoubleMu10_ZMass"),
	axisLabel = cms.string("L2 dimuon mass [GeV]"),
	mainWorkspace = cms.bool(True),
	NbinsX = cms.int32(50),
	Xmin = cms.double(60.0),
	Xmax = cms.double(160.0)
	),
    HIL3DoubleMuZMass = cms.PSet(
	pathName = cms.string("HLT_HIL3_L1DoubleMu10"),
	moduleName = cms.string("hltDoubleMuOpenL1DoubleMu10Filtered"),
	plotLabel = cms.string("HIL3DoubleMu10_ZMass"),
	axisLabel = cms.string("L3 dimuon mass [GeV]"),
	mainWorkspace = cms.bool(True),
	NbinsX = cms.int32(50),
	Xmin = cms.double(60.0),
	Xmax = cms.double(160.0)
	),
    HIL3DoubleMuJpsiMass = cms.PSet(
	pathName = cms.string("HLT_HIL3DoubleMuOpen_JpsiPsi"),
	moduleName = cms.string("hltL3fL1DoubleMuOpenL3FilteredPsi"),
	plotLabel = cms.string("HIL3DoubleMuOpen_JpsiMass"),
	axisLabel = cms.string("L3 dimuon mass [GeV]"),
	mainWorkspace = cms.bool(True),
	NbinsX = cms.int32(30),
	Xmin = cms.double(2.2),
	Xmax = cms.double(4.5)
	),
    HITktkDzeroPt = cms.PSet(
        pathName = cms.string(tracking1_pathName),
        moduleName = cms.string(tracking1_moduleName),
        plotLabel = cms.string("HITktkDzero_pT"),
        axisLabel = cms.string("dimu p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(100)
        ),
    HITktkDzeroEta = cms.PSet(
        pathName = cms.string(tracking1_pathName),
        moduleName = cms.string(tracking1_moduleName),
        plotLabel = cms.string("HITktkDzero_eta"),
        axisLabel = cms.string("dimu eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(30),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    HITktkDzeroPhi = cms.PSet(
        pathName = cms.string(tracking1_pathName),
        moduleName = cms.string(tracking1_moduleName),
        plotLabel = cms.string("HITktkDzero_Phi"),
        axisLabel = cms.string("dimu phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HITktkDzeroMass = cms.PSet(
        pathName = cms.string(tracking1_pathName),
        moduleName = cms.string(tracking1_moduleName),
        plotLabel = cms.string("HITktkDzero_Mass"),
        axisLabel = cms.string("dimu mass"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(1.7),
        Xmax = cms.double(2.0)
        ),
    HITktktkDsPt = cms.PSet(
        pathName = cms.string(tracking2_pathName),
        moduleName = cms.string(tracking2_moduleName),
        plotLabel = cms.string("HITktktkDs_pT"),
        axisLabel = cms.string("dimu p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(100)
        ),
    HITktktkDsEta = cms.PSet(
        pathName = cms.string(tracking2_pathName),
        moduleName = cms.string(tracking2_moduleName),
        plotLabel = cms.string("HITktktkDs_eta"),
        axisLabel = cms.string("dimu eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(30),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    HITktktkDsPhi = cms.PSet(
        pathName = cms.string(tracking2_pathName),
        moduleName = cms.string(tracking2_moduleName),
        plotLabel = cms.string("HITktktkDs_Phi"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HITktktkDsMass = cms.PSet(
        pathName = cms.string(tracking2_pathName),
        moduleName = cms.string(tracking2_moduleName),
        plotLabel = cms.string("HITktktkDs_Mass"),
        axisLabel = cms.string("mass"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(1.91),
        Xmax = cms.double(2.11)
        ),
    HITktktkLcPt = cms.PSet(
        pathName = cms.string(tracking3_pathName),
        moduleName = cms.string(tracking3_moduleName),
        plotLabel = cms.string("HITktktkLc_pT"),
        axisLabel = cms.string("p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(100)
        ),
    HITktktkLcEta = cms.PSet(
        pathName = cms.string(tracking3_pathName),
        moduleName = cms.string(tracking3_moduleName),
        plotLabel = cms.string("HITktktkLc_eta"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(30),
        Xmin = cms.double(-2.4),
        Xmax = cms.double(2.4)
        ),
    HITktktkLcPhi = cms.PSet(
        pathName = cms.string(tracking3_pathName),
        moduleName = cms.string(tracking3_moduleName),
        plotLabel = cms.string("HITktktkLc_Phi"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
        ),
    HITktktkLcMass = cms.PSet(
        pathName = cms.string(tracking3_pathName),
        moduleName = cms.string(tracking3_moduleName),
        plotLabel = cms.string("HITktktkLc_Mass"),
        axisLabel = cms.string("mass"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(2.1),
        Xmax = cms.double(2.5)
        ),
    HIFullTracksMultiplicity6080Pt = cms.PSet(
        pathName = cms.string(fulltrack_pathName),
        moduleName = cms.string(fulltrack_moduleName),
        plotLabel = cms.string("HIFullTracksMultiplicity6080_pT"),
        axisLabel = cms.string("p_{T} [GeV]"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(0),
        Xmax = cms.double(20)
        ),
    HIFullTracksMultiplicity6080Eta = cms.PSet(
        pathName = cms.string(fulltrack_pathName),
        moduleName = cms.string(fulltrack_moduleName),
        plotLabel = cms.string("HIFullTracksMultiplicity6080_eta"),
        axisLabel = cms.string("eta"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-2.5),
        Xmax = cms.double(2.5)
        ),
    HIFullTracksMultiplicity6080Phi = cms.PSet(
        pathName = cms.string(fulltrack_pathName),
        moduleName = cms.string(fulltrack_moduleName),
        plotLabel = cms.string("HIFullTracksMultiplicity6080_Phi"),
        axisLabel = cms.string("phi"),
        mainWorkspace = cms.bool(True),
        NbinsX = cms.int32(50),
        Xmin = cms.double(-3.2),
        Xmax = cms.double(3.2)
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
