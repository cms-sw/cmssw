import FWCore.ParameterSet.Config as cms

hltMuonOfflineAnalyzer = cms.EDAnalyzer("HLTMuonOfflineAnalyzer",

    hltProcessName = cms.string("HLT"),
    destination = cms.untracked.string("HLT/Muon/Distributions/globalMuons"),

    # HLT paths passing any one of these regular expressions will be included
    hltPathsToCheck = cms.vstring(
        "HLT_(L[12])?(Double)?(Iso)?Mu[0-9]*(Open)?(_NoVertex)?(_v[0-9]*)?$",
    ),

    ## All input tags are specified in this pset for convenience
    inputTags = cms.PSet(
        recoMuon       = cms.InputTag("muons"),
        beamSpot       = cms.InputTag("offlineBeamSpot"),
        triggerSummary = cms.InputTag("hltTriggerSummaryAOD"),
        triggerResult  = cms.InputTag("TriggerResults","","HLT"),
    ),

    ## Both 1D and 2D plots use the binnings defined here
    binParams = cms.untracked.PSet(
        ## parameters for fixed-width plots
        eta        = cms.untracked.vdouble( 20,  -2.40,   2.40),
        phi        = cms.untracked.vdouble( 20,  -3.14,   3.14),
        z0         = cms.untracked.vdouble( 10, -15.00,  15.00),
        d0         = cms.untracked.vdouble( 10,  -0.50,   0.50),
        zMass      = cms.untracked.vdouble(100,  65.00, 115.00),
        charge     = cms.untracked.vdouble(  2,  -2.00,   2.00),
        resolution = cms.untracked.vdouble( 20,  -0.15,   0.15),
        deltaR     = cms.untracked.vdouble( 20,   0.00,   0.10),
        phiCoarse  = cms.untracked.vdouble( 10,  -3.14,   3.14),
        ## parameters for variable-width plots
        etaCoarse = cms.untracked.vdouble(-2.4, -2.1, -1.6, -1.2, -0.8, 0.0,
                                           0.8,  1.2,  1.6,  2.1,  2.4),
        pt = cms.untracked.vdouble(  0.0,   2.0,   4.0, 
                                     6.0,   8.0,  10.0, 
                                    20.0,  30.0,  40.0, 
                                   100.0, 200.0, 400.0),
    ),

    deltaRCuts = cms.PSet(
        L1 = cms.untracked.double(0.30),
        L2 = cms.untracked.double(0.30),
        L3 = cms.untracked.double(0.05),
    ),

    # This collection is used to fill most distributions
    targetParams = cms.PSet(
        requiredTriggers = cms.untracked.vstring(""),
        # The d0 and z0 cuts are required for the inner track of the
        # reco muons, taken with respect to the beamspot
        d0Cut = cms.untracked.double(2.0),
        z0Cut = cms.untracked.double(25.0),
        # cuts
        recoCuts = cms.untracked.string("isGlobalMuon && abs(eta) < 2.4"),
        hltCuts  = cms.untracked.string("abs(eta) < 2.4"),
    ),

    # If this PSet is empty, then no "tag and probe" plots are produced;
    # the cuts used for the tags are specified by targetParams
    probeParams = cms.PSet(
        requiredTriggers = cms.untracked.vstring(""),
        # The d0 and z0 cuts are required for the inner track of the
        # reco muons, taken with respect to the beamspot
        d0Cut = cms.untracked.double(2.0),
        z0Cut = cms.untracked.double(25.0),
        # cuts
        recoCuts = cms.untracked.string("isGlobalMuon && abs(eta) < 2.0"),
        hltCuts  = cms.untracked.string("abs(eta) < 2.0"),
    ),

)
