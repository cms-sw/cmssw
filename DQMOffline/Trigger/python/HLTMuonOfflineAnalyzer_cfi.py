import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hltMuonOfflineAnalyzer = DQMEDAnalyzer('HLTMuonOfflineAnalyzer',

    ## Used when fetching triggerSummary and triggerResults
    hltProcessName = cms.string("HLT"),

    ## Location of plots in DQM
    destination = cms.untracked.string("HLT/Muon/Distributions/globalMuons"),

    ## HLT paths passing any one of these regular expressions will be included
    hltPathsToCheck = cms.vstring(
      "HLT_Mu8_TrkIsoVVL_v",
      "HLT_Mu50_v",
      "HLT_Mu24_v",
      "HLT_IsoMu24_v",
      "HLT_IsoMu27_v",
      "HLT_IsoMu20_v",
      "HLT_HIL3Mu12_v", #for HI
      "HLT_HIL3Mu15_v", #for HI
      "HLT_HIL3Mu20_v", #for HI
      "HLT_CascadeMu100_v",
      "HLT_HighPtTkMu100_v"
    ),

#HLT_Mu15_eta2p1_TriCentral_40_20_20_BTagIP3D1stTrack_v3 matches HLT_Mu15_eta2p1_v

    ## All input tags are specified in this pset for convenience
    inputTags = cms.PSet(
        recoMuon       = cms.InputTag("muons"),
        beamSpot       = cms.InputTag("offlineBeamSpot"),
        offlinePVs     = cms.InputTag("offlinePrimaryVertices"),
        triggerSummary = cms.InputTag("hltTriggerSummaryAOD"),
        triggerResults = cms.InputTag("TriggerResults")
    ),

    ## Both 1D and 2D plots use the binnings defined here
    binParams = cms.untracked.PSet(
        ## parameters for fixed-width plots
        NVertex    = cms.untracked.vdouble( 1,10,15,20,25,30,35,40,45,50,55,60,65,70,100),
        NVertexFine= cms.untracked.vdouble( 1,5,10,12.5,15,17.5,20,22.5,25,27.5,30,32.5,35,37.5,40,42.5,45,47.5,50,52.5,55,57.5,60,62.5,65,67.5,70,85,100),
        eta        = cms.untracked.vdouble( 20,  -2.40,   2.40),
        phi        = cms.untracked.vdouble( 20,  -3.14,   3.14),
        phiHEP17   = cms.untracked.vdouble( -3.14,-2.4,-1.8,-1.0,-0.4,0.0,0.4,1.0,1.8,2.4,3.14),
        z0         = cms.untracked.vdouble( 10, -0.15,  0.15),
        z0Fine     = cms.untracked.vdouble( 20, -0.15,  0.15),
        d0         = cms.untracked.vdouble( 10,  -0.50,   0.50),
        zMass      = cms.untracked.vdouble( 50,  65.00, 115.00),
        jpsiMass   = cms.untracked.vdouble( 60,   0.00,   6.00),
        charge     = cms.untracked.vdouble(  2,  -2.00,   2.00),
        deltaR     = cms.untracked.vdouble( 20,   0.00,   0.05),
        deltaR2    = cms.untracked.vdouble( 20,    0.0,   4.5 ),
        phiCoarse  = cms.untracked.vdouble( 10,  -3.14,   3.14),
        resolutionRel = cms.untracked.vdouble( 40,  -0.30,   0.30),
        resolutionEta = cms.untracked.vdouble( 20,  -0.01,   0.01),
        resolutionPhi = cms.untracked.vdouble( 20,  -0.01,   0.01),
        ## parameters for variable-width plots
        etaCoarse = cms.untracked.vdouble(-2.4, -2.1, -1.6, -1.2, -0.8, 0.0,
                                           0.8,  1.2,  1.6,  2.1,  2.4),
        etaFine = cms.untracked.vdouble(-2.4,-2.1,-1.6,-1.2,-0.9,-0.3,
                                         -0.2,0.2,0.3,0.9,1.2,1.6,2.1,2.4),
        phiFine = cms.untracked.vdouble(-3.14,-(11.0/12.0)*3.14,-(9.0/12.0)*3.14,-(7.0/12.0)*3.14,-(5.0/12.0)*3.14,-
(3.0/12.0)*3.14,-(1.0/12.0)*3.14,(1.0/12.0)*3.14,(3.0/12.0)*3.14,(5.0/12.0)*3.14,(7.0/12.0)*3.14,
(9.0/12.0)*3.14,(11.0/12.0)*3.14,3.14),
        ptCoarse = cms.untracked.vdouble(10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0),
        ptFine   = cms.untracked.vdouble(10.0,15.0, 20.0,30.0, 40.0,50.0, 60.0,70.0, 80.0,90.0, 100.0,150., 200.0),
        pt = cms.untracked.vdouble(  0.0,   2.0,   4.0, 
                                     6.0,   8.0,  10.0, 
                                    20.0,  30.0,  40.0, 
                                   100.0, 200.0, 400.0),
    ),

    ## These parameters define which objects are used to fill plots
    plotCuts = cms.PSet(
        ## not applied on eta plots
        maxEta = cms.untracked.double(2.10),
        ## only fill plots for muons with pt > ceil(hltThreshold * minPtFactor)
        ## ex: for HLT_Mu17, ceil(17 * 1.2 ) = 21, so we require pT > 21
        minPtFactor = cms.untracked.double(1.20),
        ## deltaR cuts
        L1DeltaR = cms.untracked.double(0.30),
        L2DeltaR = cms.untracked.double(0.30),
        L3DeltaR = cms.untracked.double(0.05),
    ),

    ## Only events passing all these triggers will be considered
    requiredTriggers   = cms.untracked.vstring(),

    ## This collection is used to fill most distributions
    targetParams = cms.PSet(
        ## The d0 and z0 cuts are required for the inner track of the
        ## reco muons, taken with respect to the beamspot
        d0Cut = cms.untracked.double(2.0),
        z0Cut = cms.untracked.double(25.0),
        ## cuts
        recoCuts = cms.untracked.string("isGlobalMuon && abs(eta) < 2.4"),
        hltCuts  = cms.untracked.string("abs(eta) < 2.4"),
    ),

    ## If this PSet is empty, then no "tag and probe" plots are produced;
    ## the cuts used for the tags are specified by targetParams
    probeParams = cms.PSet(
        ## The d0 and z0 cuts are required for the inner track of the
        ## reco muons, taken with respect to the beamspot
        d0Cut = cms.untracked.double(2.0),
        z0Cut = cms.untracked.double(25.0),
        ## cuts
        recoCuts = cms.untracked.string("isGlobalMuon && abs(eta) < 2.4"),
        hltCuts  = cms.untracked.string("abs(eta) < 2.4"),
    ),

)
