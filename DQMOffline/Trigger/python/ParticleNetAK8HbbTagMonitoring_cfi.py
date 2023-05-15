import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.ParticleNetJetTagMonitor_cfi import ParticleNetJetTagMonitor as _particleNetJetTagMonitor

particleNetAK8HbbTagMonitoring = _particleNetJetTagMonitor.clone(
    ## general options
    FolderName = "HLT/HIG/PNETAK8/HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35_or_HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40/",
    requireValidHLTPaths = True,
    requireHLTOfflineJetMatching = True,
    ## objects
    muons = "muons",
    electrons = "gedGsfElectrons",
    jets = "ak8PFJetsPuppi",
    jetPNETScore = "pfMassDecorrelatedParticleNetDiscriminatorsJetTags:XbbvsQCD",
    jetPNETScoreHLT = "hltParticleNetDiscriminatorsJetTagsAK8:HbbVsQCD",
    jetsForHTandBTag  = "ak4PFJetsPuppi",
    jetPNETScoreForHTandBTag = "pfParticleNetAK4DiscriminatorsJetTagsForRECO:BvsAll",
    jetSoftDropMass = "ak8PFJetsPuppiSoftDropMass",
    met = "pfMetPuppi",
    jecForMC = "",
    jecForData = "",
    ## PV selection
    vertexSelection = "!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2",
    ## Muon selection (based on the definition of https://cmssdt.cern.ch/lxr/source/DataFormats/MuonReco/interface/Muon.h)
    tagMuonSelection = "pt > 60 && abs(eta) < 2.4 && passed(4) && passed(256)",
    vetoMuonSelection = "pt > 10 && abs(eta) < 2.4 && passed(1) && passed(128)",
    maxLeptonDxyCut = 0.1,
    maxLeptonDzCut = 0.2,
    ntagmuons = -1,
    nvetomuons = -1,
    ## Electron selection
    tagElectronSelection = "pt > 60 && abs(eta) < 2.5",
    vetoElectronSelection = "pt > 15 && abs(eta) < 2.5",
    tagElectronID = "egmGsfElectronIDsForDQM:cutBasedElectronID-Fall17-94X-V1-tight",
    vetoElectronID = "egmGsfElectronIDsForDQM:cutBasedElectronID-Fall17-94X-V1-loose",
    ntagelectrons = -1,
    nvetoelectrons = -1,
    ## Lepton counting
    ntagleptons = 1,
    nvetoleptons = 1,
    ## Emu pairs
    dileptonSelection = "",
    nemupairs = -1,
    ## Jet AK8 selection
    jetSelection = "pt > 180 && abs(eta) < 2.4",
    minPNETScoreCut = 0.1,
    minSoftDropMassCut = 50,
    maxSoftDropMassCut = 110,
    njets = 1,
    ## B-tagged jet selection
    jetSelectionForHTandBTag = "pt > 30 && abs(eta) < 2.5",
    minPNETBTagCut = 0.25,
    nbjets = 2,
    ## PF-MET selection
    metSelection = "pt > 30",
    ## Jet lepton cleaning
    lepJetDeltaRmin = 0.8,
    lepJetDeltaRminForHTandBTag = 0.4,
    ## Trigger matching
    hltRecoDeltaRmax = 0.8,
    ntrigobjecttomatch = 1,
    ## binning for efficiency
    NjetBinning = [0,1,2,3,4,5,6,7,8],
    HTBinning = [100,200,300,400,500,600,700,800,1000,1250],
    leptonPtBinning = [75,100,125,150,200,250,300,400,600],
    leptonEtaBinning = [-2.5,-2.0,-1.5,-0.5,0.,0.5,1.0,1.5,2.0,2.5],
    diLeptonPtBinning = [],
    diLeptonMassBinning = [],
    jet1PtBinning = [180,200,220,240,260,280,300,325,350,375,400,450,500,600,750,900],
    jet2PtBinning = [],
    jet1EtaBinning = [-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5],
    jet2EtaBinning = [],
    jet1PNETscoreBinning = [0,0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    jet2PNETscoreBinning = [],
    jet1PNETscoreTransBinning = [0,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.4,0.6,0.8,1.0,2,3,6],
    jet2PNETscoreTransBinning = [],
    jet1PtBinning2d = [180,220,260,300,340,400,500,750,1250],
    jet2PtBinning2d = [],
    jet1EtaBinning2d = [-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5],
    jet2EtaBinning2d = [],
    jet1PNETscoreBinning2d = [0,0.03,0.06,0.08,0.12,0.18,0.25,0.35,0.50,0.70,0.90,1.0],
    jet2PNETscoreBinning2d = [],
    jet1PNETscoreTransBinning2d = [0,0.03,0.06,0.08,0.12,0.18,0.25,0.40,0.60,0.85,1.5,3,6],
    jet2PNETscoreTransBinning2d = [],
    ## trigger for numerator and denominator
    numGenericTriggerEvent = dict(
        hltPaths      = ["HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PNetBB0p06_v*", "HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PNetBB0p06_v*",],
        andOr         = False,
        andOrHlt      = True,
        #hltInputTag   = "TriggerResults::reHLT", ## when testing in the DQM workflow (https://twiki.cern.ch/twiki/bin/viewauth/CMS/HLTValidationAndDQM)
        hltInputTag   = "TriggerResults::HLT",
        errorReplyHlt = False,
        dcsInputTag   = "scalersRawToDigi",
        dcsPartitions = [24, 25, 26, 27, 28, 29],
        andOrDcs      = False,
        errorReplyDcs = True,
        verbosityLevel = 1,
    ),
    denGenericTriggerEvent = dict(
        hltPaths      = ["HLT_IsoMu50_AK8PFJet230_SoftDropMass40_v*",
                         "HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_v*",
                     ],
        andOr         = False,
        andOrHlt      = True,
        #hltInputTag   = "TriggerResults::reHLT", ## when testing in the DQM workflow (https://twiki.cern.ch/twiki/bin/viewauth/CMS/HLTValidationAndDQM)
        hltInputTag   = "TriggerResults::HLT",
        errorReplyHlt = False,
        dcsInputTag   = "scalersRawToDigi",
        dcsPartitions = [24, 25, 26, 27, 28, 29],
        andOrDcs      = False,
        errorReplyDcs = True,
        verbosityLevel = 1,
    ),
)
