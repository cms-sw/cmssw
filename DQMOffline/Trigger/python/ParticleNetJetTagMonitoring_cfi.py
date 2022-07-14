import FWCore.ParameterSet.Config as cms
import math
from DQMOffline.Trigger.ParticleNetJetTagMonitor_cfi import ParticleNetJetTagMonitor

ParticleNetAK4BTagMonitoring = ParticleNetJetTagMonitor.clone(
    ## general options
    FolderName = "HLT/HIG/PNETAK4/HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBTagParticleNet_2BTagSum0p65/",
    requireValidHLTPaths = True,
    requireHLTOfflineJetMatching = True,
    ## objects
    muons = "muons",
    electrons = "gedGsfElectrons",
    jets = "ak4PFJetsCHS",
    jetPNETScore = "pfParticleNetAK4DiscriminatorsJetTagsForRECO:BvsAll",
    jetPNETScoreHLT = "hltParticleNetDiscriminatorsJetTags:BvsAll",    
    jetsForHTandBTag = "",
    jetPNETScoreForHTandBTag = "",
    jetSoftDropMass = "",
    met = "pfMetPuppi",
    jecForMC = "ak4PFCHSL1FastL2L3Corrector",
    jecForData = "ak4PFCHSL1FastL2L3ResidualCorrector",
    ## PV selection
    vertexSelection = "!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2",
    ## Muon selection (based on the definition of https://cmssdt.cern.ch/lxr/source/DataFormats/MuonReco/interface/Muon.h)    
    tagMuonSelection = "pt > 15 && abs(eta) < 2.4 && passed(4) && passed(256)",
    vetoMuonSelection = "pt > 10 && abs(eta) < 2.4 && passed(1) && passed(128)",
    maxLeptonDxyCut = 0.1,
    maxLeptonDzCut = 0.2,
    ntagmuons = 1,
    nvetomuons = 1,
    ## Electron selection
    tagElectronSelection = "pt > 25 && abs(eta) < 2.5",
    vetoElectronSelection = "pt > 15 && abs(eta) < 2.5",
    tagElectronID = "egmGsfElectronIDsForDQM:cutBasedElectronID-Fall17-94X-V1-tight",
    vetoElectronID = "egmGsfElectronIDsForDQM:cutBasedElectronID-Fall17-94X-V1-loose",
    ntagelectrons = 1,
    nvetoelectrons = 1,
    ## Total number of leptons (electron+muons) in the event
    ntagleptons = 2,
    nvetoleptons = 2,
    ## Emu pairs
    dileptonSelection = "mass > 20 && charge == 0",
    nemupairs = 1,
    ## jet selection (main jet collection)
    jetSelection = "pt > 30 && abs(eta) < 2.5",
    minPNETScoreCut = 0.1,
    minSoftDropMassCut = 0,
    maxSoftDropMassCut = 10000,
    njets  = 2,
    ## Bjet selection (ak4 jets)
    jetSelectionForHTandBTag = "pt > 30 && abs(eta) < 2.5",
    nbjets = -1,
    ## Met selection
    metSelection = "pt > 0",
    ## Cleaning jet-lepton
    lepJetDeltaRmin = 0.4,
    ## Match reco with HLT
    hltRecoDeltaRmax = 0.4,        
    ntrigobjecttomatch = 2,
    ## binning for efficiecny
    NjetBinning = [0,1,2,3,4,5,6,7,8],
    HTBinning = [0,50,100,150,200,250,300,350,400,500,600,750,1000],
    leptonPtBinning = [20,30,40,50,60,75,90,110,135,175,225,300],
    leptonEtaBinning = [-2.5,-2.25,-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5],
    diLeptonPtBinning = [0,20,30,40,50,60,75,90,110,135,175,225,300],
    diLeptonMassBinning = [20,30,40,50,60,70,80,90,100,110,120,130,140,155,170,185,200,220,240,260,300],
    jet1PtBinning = [20,30,40,50,60,75,90,110,130,150,175,200,225,250,300,400,500],
    jet2PtBinning = [20,30,40,50,60,75,90,110,130,150,175,200,225,250,300],
    jet1EtaBinning = [-2.5,-2.25,-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5],
    jet2EtaBinning = [-2.5,-2.25,-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5],
    jet1PNETscoreBinning = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.925,0.95,0.975,1],
    jet2PNETscoreBinning = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.925,0.95,0.975,1],
    jet1PNETscoreTransBinning = [0,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.25,1.50,1.75,2.25,2.75,4],
    jet2PNETscoreTransBinning = [0,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.25,1.50,1.75,2.25,2.75,4],
    jet1PtBinning2d = [20,30,40,50,75,100,150,250,500],
    jet2PtBinning2d = [20,30,40,50,75,100,125,150,250],
    jet1EtaBinning2d = [-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5],
    jet2EtaBinning2d = [-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5],
    jet1PNETscoreBinning2d = [0,0.15,0.30,0.45,0.60,0.75,0.85,0.90,0.95,0.975,1],
    jet2PNETscoreBinning2d = [0,0.15,0.30,0.45,0.60,0.75,0.85,0.90,0.95,0.975,1],
    jet1PNETscoreTransBinning2d = [0,0.15,0.30,0.45,0.60,0.75,1,1.5,2,2.5,4],
    jet2PNETscoreTransBinning2d = [0,0.15,0.30,0.45,0.60,0.75,1,1.5,2,2.5,4],
    ## trigger for numerator and denominator
    numGenericTriggerEvent = dict(
        hltPaths      = ["HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_PFBTagParticleNet_2BTagSum0p65_v*"],
        andOr         = False,
        andOrHlt      = True,
        #hltInputTag   = "TriggerResults::reHLT", ## when testing in the DQM workflow (https://twiki.cern.ch/twiki/bin/viewauth/CMS/HLTValidationAndDQM)
        hltInputTag   = "TriggerResults::HLT",
        errorReplyHlt = False,
        dcsInputTag   = "scalersRawToDigi",
        dcsPartitions = [24, 25, 26, 27, 28, 29],
        andOrDcs      = False,
        errorReplyDcs = True,
        verbosityLevel = 1),
    denGenericTriggerEvent = dict(
        hltPaths      = ["HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ_PFDiJet30_v*"],
        andOr         = False,
        andOrHlt      = True,
        #hltInputTag   = "TriggerResults::reHLT", ## when testing in the DQM workflow (https://twiki.cern.ch/twiki/bin/viewauth/CMS/HLTValidationAndDQM)
        hltInputTag   = "TriggerResults::HLT",
        errorReplyHlt = False,
        dcsInputTag   = "scalersRawToDigi",
        dcsPartitions = [24, 25, 26, 27, 28, 29],
        andOrDcs      = False,
        errorReplyDcs = True,
        verbosityLevel = 1),
 )

 
ParticleNetAK8HbbTagMonitoring = ParticleNetJetTagMonitor.clone(
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
    jetsForHTandBTag  = "ak4PFJetsCHS",
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
    jet1PNETscoreBinning = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.925,0.95,0.975,1],
    jet2PNETscoreBinning = [],
    jet1PNETscoreTransBinning = [0,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.25,1.50,1.75,2.25,2.75,4],
    jet2PNETscoreTransBinning = [],
    jet1PtBinning2d = [180,220,260,300,340,400,500,750,1250],
    jet2PtBinning2d = [],
    jet1EtaBinning2d = [-2.5,-2.0,-1.5,-1.0,-0.5,0.,0.5,1.0,1.5,2.0,2.5],
    jet2EtaBinning2d = [],
    jet1PNETscoreBinning2d = [0,0.15,0.30,0.45,0.60,0.75,0.85,0.90,0.95,0.975,1],
    jet2PNETscoreBinning2d = [],
    jet1PNETscoreTransBinning2d = [0,0.15,0.30,0.45,0.60,0.75,1,1.5,2,2.5,4],
    jet2PNETscoreTransBinning2d = [],
    ## trigger for numerator and denominator
    numGenericTriggerEvent = dict(
        hltPaths      = ["HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35_v*",
                         "HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35_v*"
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
        verbosityLevel = 1),
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
        verbosityLevel = 1),
)
