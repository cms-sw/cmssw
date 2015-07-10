##########################################################
##          MONOX COMMON MODULES ARE DEFINED HERE       ##
## skimming modules are configured to not cut anything  ##
##########################################################

import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.Heppy.analyzers.core.all import *
from PhysicsTools.Heppy.analyzers.objects.all import *
from PhysicsTools.Heppy.analyzers.gen.all import *
import os

from CMGTools.TTHAnalysis.analyzers.ttHhistoCounterAnalyzer import ttHhistoCounterAnalyzer
dmCounter = cfg.Analyzer(
    ttHhistoCounterAnalyzer, name="ttHhistoCounterAnalyzer",
    )

PDFWeights = []
#PDFWeights = [ ("CT10",53), ("MSTW2008lo68cl",41), ("NNPDF21_100",101) ]

# Find the initial events before the skim
skimAnalyzer = cfg.Analyzer(
    SkimAnalyzerCount, name='skimAnalyzerCount',
    useLumiBlocks = False,
    )

# Pick individual events (normally not in the path)
eventSelector = cfg.Analyzer(
    EventSelector,name="EventSelector",
    toSelect = []  # here put the event numbers (actual event numbers from CMSSW)
    )

# Apply json file (if the dataset has one)
jsonAna = cfg.Analyzer(
    JSONAnalyzer, name="JSONAnalyzer",
    )

# Filter using the 'triggers' and 'vetoTriggers' specified in the dataset
triggerAna = cfg.Analyzer(
    TriggerBitFilter, name="TriggerBitFilter",
    )

# Create flags for trigger bits
triggerFlagsAna = cfg.Analyzer(
    TriggerBitAnalyzer, name="TriggerFlags",
    processName = 'HLT',
    triggerBits = {
        # "<name>" : [ 'HLT_<Something>_v*', 'HLT_<SomethingElse>_v*' ] 
    }
    )
# Create flags for MET filter bits
eventFlagsAna = cfg.Analyzer(
    TriggerBitAnalyzer, name="EventFlags",
    processName = 'PAT',
    outprefix   = 'Flag',
    triggerBits = {
        "HBHENoiseFilter" : [ "Flag_HBHENoiseFilter" ],
        "CSCTightHaloFilter" : [ "Flag_CSCTightHaloFilter" ],
        "hcalLaserEventFilter" : [ "Flag_hcalLaserEventFilter" ],
        "EcalDeadCellTriggerPrimitiveFilter" : [ "Flag_EcalDeadCellTriggerPrimitiveFilter" ],
        "goodVertices" : [ "Flag_goodVertices" ],
        "trackingFailureFilter" : [ "Flag_trackingFailureFilter" ],
        "eeBadScFilter" : [ "Flag_eeBadScFilter" ],
        "ecalLaserCorrFilter" : [ "Flag_ecalLaserCorrFilter" ],
        "trkPOGFilters" : [ "Flag_trkPOGFilters" ],
        "trkPOG_manystripclus53X" : [ "Flag_trkPOG_manystripclus53X" ],
        "trkPOG_toomanystripclus53X" : [ "Flag_trkPOG_toomanystripclus53X" ],
        "trkPOG_logErrorTooManyClusters" : [ "Flag_trkPOG_logErrorTooManyClusters" ],
        "METFilters" : [ "Flag_METFilters" ],
    }
    )

# Select a list of good primary vertices (generic)
vertexAna = cfg.Analyzer(
    VertexAnalyzer, name="VertexAnalyzer",
    vertexWeight = None,
    fixedWeight = 1,
    verbose = False
    )


# This analyzer actually does the pile-up reweighting (generic)
pileUpAna = cfg.Analyzer(
    PileUpAnalyzer, name="PileUpAnalyzer",
    true = True,  # use number of true interactions for reweighting
    makeHists=False
    )


## Gen Info Analyzer (generic, but should be revised)
genAna = cfg.Analyzer(
    GeneratorAnalyzer, name="GeneratorAnalyzer",
    # BSM particles that can appear with status <= 2 and should be kept
    stableBSMParticleIds = [ 1000022 ],
    # Particles of which we want to save the pre-FSR momentum (a la status 3).
    # Note that for quarks and gluons the post-FSR doesn't make sense,
    # so those should always be in the list
    savePreFSRParticleIds = [ 1,2,3,4,5, 11,12,13,14,15,16, 21 ],
    # Make also the list of all genParticles, for other analyzers to handle
    makeAllGenParticles = True,
    # Make also the splitted lists
    makeSplittedGenLists = True,
    allGenTaus = False,
    # Save LHE weights from LHEEventProduct
    makeLHEweights = True,
    # Print out debug information
    verbose = False,
    )

genHiggsAna = cfg.Analyzer(
    HiggsDecayModeAnalyzer, name="HiggsDecayModeAnalyzer",
    filterHiggsDecays = False,
)
genHFAna = cfg.Analyzer(
    GenHeavyFlavourAnalyzer, name="GenHeavyFlavourAnalyzer",
    status2Only = False,
    bquarkPtCut = 15.0,
)
pdfwAna = cfg.Analyzer(
    PDFWeightsAnalyzer, name="PDFWeightsAnalyzer",
    PDFWeights = [ pdf for pdf,num in PDFWeights ],
    doPDFVars = True,
    )

# Lepton Analyzer (generic)
lepAna = cfg.Analyzer(
    LeptonAnalyzer, name="leptonAnalyzer",
    # input collections
    muons='slimmedMuons',
    electrons='slimmedElectrons',
    rhoMuon= 'fixedGridRhoFastjetAll',
    rhoElectron = 'fixedGridRhoFastjetAll',
    # energy scale corrections and ghost muon suppression (off by default)
    doMuScleFitCorrections=False, # "rereco"
    doRochesterCorrections=False,
    doElectronScaleCorrections=False, # "embedded" in 5.18 for regression
    doSegmentBasedMuonCleaning=False,
    # inclusive very loose muon selection
    inclusive_muon_id  = "POG_ID_Loose",
    inclusive_muon_pt  = 3,
    inclusive_muon_eta = 2.4,
    inclusive_muon_dxy = 0.5,
    inclusive_muon_dz  = 1.0,
    muon_dxydz_track = "innerTrack",
    # veto muon selection
    loose_muon_id     = "POG_ID_Loose",
    loose_muon_pt     = 10,
    loose_muon_eta    = 2.4,
    loose_muon_dxy    = 0.2,
    loose_muon_dz     = 0.5,
    loose_muon_relIso = 0.4,
    # inclusive very loose electron selection
    inclusive_electron_id  = "",
    inclusive_electron_pt  = 5,
    inclusive_electron_eta = 2.5,
    inclusive_electron_dxy = 0.5,
    inclusive_electron_dz  = 1.0,
    inclusive_electron_lostHits = 5.0,
    # veto electron selection
    loose_electron_id     = "POG_Cuts_ID_PHYS14_25ns_v2_ConvVetoDxyDz_Veto_full5x5",
    loose_electron_pt     = 10,
    loose_electron_eta    = 2.5,
    loose_electron_dxy    = 0.5,
    loose_electron_dz     = 1.0,
    loose_electron_relIso = 1.0,
    loose_electron_lostHits = 5.0,
    # muon isolation correction method (can be "rhoArea" or "deltaBeta")
    mu_isoCorr = "deltaBeta" ,
    mu_effectiveAreas = "Phys14_25ns_v1", #(can be 'Data2012' or 'Phys14_25ns_v1')
    # electron isolation correction method (can be "rhoArea" or "deltaBeta")
    ele_isoCorr = "deltaBeta" ,
    el_effectiveAreas = "Phys14_25ns_v1" , #(can be 'Data2012' or 'Phys14_25ns_v1')
    ele_tightId = "Cuts_2012" ,
    # Mini-isolation, with pT dependent cone: will fill in the miniRelIso, miniRelIsoCharged, miniRelIsoNeutral variables of the leptons (see https://indico.cern.ch/event/368826/ )
    doMiniIsolation = False, # off by default since it requires access to all PFCandidates 
    packedCandidates = 'packedPFCandidates',
    miniIsolationPUCorr = 'rhoArea', # Allowed options: 'rhoArea' (EAs for 03 cone scaled by R^2), 'deltaBeta', 'raw' (uncorrected), 'weights' (delta beta weights; not validated)
    miniIsolationVetoLeptons = None, # use 'inclusive' to veto inclusive leptons and their footprint in all isolation cones
    # minimum deltaR between a loose electron and a loose muon (on overlaps, discard the electron)
    min_dr_electron_muon = 0.05,
    # do MC matching 
    do_mc_match = True, # note: it will in any case try it only on MC, not on data
    match_inclusiveLeptons = False, # match to all inclusive leptons
    )


## MET-based Skim
from CMGTools.MonoXAnalysis.analyzers.monoJetSkimmer import monoJetSkimmer
monoJetSkim = cfg.Analyzer(
    monoJetSkimmer, name='monoJetSkimmer',
    jets      = "cleanJets", # jet collection to use
    jetPtCuts = [],          # e.g. [60,40,30,20] to require at least four jets with pt > 60,40,30,20 
    metCut = 0               # MET cut      
    )

## number of leptons Skim
from CMGTools.MonoXAnalysis.analyzers.monoJetCtrlLepSkimmer import monoJetCtrlLepSkimmer
monoJetCtrlLepSkim = cfg.Analyzer(
    monoJetCtrlLepSkimmer, name='monoJetCtrlLepSkimmer',
    minLeptons = 1,
    maxLeptons = 999,
    #idCut  = "lepton.relIso03 < 0.2" # can give a cut
    idCut = 'lepton.muonID("POG_ID_Medium") if abs(lepton.pdgId())==13 else lepton.electronID("POG_Cuts_ID_PHYS14_25ns_v1_Veto_full5x5")',
    ptCuts = [20],                # can give a set of pt cuts on the leptons
    )

## Photon Analyzer (generic)
photonAna = cfg.Analyzer(
    PhotonAnalyzer, name='photonAnalyzer',
    photons='slimmedPhotons',
    ptMin = 15,
    etaMax = 2.5,
    gammaID = "POG_PHYS14_25ns_Loose",
    rhoPhoton = 'fixedGridRhoFastjetAll',
    gamma_isoCorr = 'rhoArea',
    do_mc_match = True,
    do_randomCone = False,
)

## Tau Analyzer (generic)
tauAna = cfg.Analyzer(
    TauAnalyzer, name="tauAnalyzer",
    # inclusive very loose hadronic tau selection
    inclusive_ptMin = 18,
    inclusive_etaMax = 9999,
    inclusive_dxyMax = 1000.,
    inclusive_dzMax = 1000,
    inclusive_vetoLeptons = False,
    inclusive_leptonVetoDR = 0.4,
    inclusive_decayModeID = "", # ignored if not set or ""
    inclusive_tauID = "decayModeFinding",
    inclusive_vetoLeptonsPOG = False, # If True, the following two IDs are required
    inclusive_tauAntiMuonID = "",
    inclusive_tauAntiElectronID = "",
    # loose hadronic tau selection
    loose_ptMin = 18,
    loose_etaMax = 9999,
    loose_dxyMax = 1000.,
    loose_dzMax = 1000,
    loose_vetoLeptons = False,
    loose_leptonVetoDR = 0.4,
    loose_decayModeID = "decayModeFinding", # ignored if not set or ""
    loose_tauID = "decayModeFinding",
    loose_vetoLeptonsPOG = False, # If True, the following two IDs are required
    loose_tauAntiMuonID = "",
    loose_tauAntiElectronID = "",
    loose_tauLooseID = "decayModeFinding"
)

##------------------------------------------
###  ISOLATED TRACK
###------------------------------------------                                                                                                                                                                
#
## those are the cuts for the nonEMu                                                                                                                                                                         
isoTrackAna = cfg.Analyzer(
    IsoTrackAnalyzer, name='isoTrackAnalyzer',
    setOff=False,
    #####
    candidates='packedPFCandidates',
    candidatesTypes='std::vector<pat::PackedCandidate>',
    ptMin = 5, # for pion 
    ptMinEMU = 5, # for EMU
    dzMax = 0.1,
    #####
    isoDR = 0.3,
    ptPartMin = 0,
    dzPartMax = 0.1,
    maxAbsIso = 8,
    #####
    MaxIsoSum = 0.1, ### unused
    MaxIsoSumEMU = 0.2, ### unused
    doSecondVeto = False,
    #####
    doPrune = True
    )

## Jets Analyzer (generic)
jetAna = cfg.Analyzer(
    JetAnalyzer, name='jetAnalyzer',
    jetCol = 'slimmedJets',
    copyJetsByValue = False,      #Whether or not to copy the input jets or to work with references (should be 'True' if JetAnalyzer is run more than once)
    genJetCol = 'slimmedGenJets',
    rho = ('fixedGridRhoFastjetAll','',''),
    jetPt = 25.,
    jetEta = 4.7,
    jetEtaCentral = 2.5,
    jetLepDR = 0.4,
    jetLepArbitration = (lambda jet,lepton : (jet,lepton)), # you can decide which to keep in case of overlaps; e.g. if the jet is b-tagged you might want to keep the jet
    cleanSelectedLeptons = False, #Whether to clean 'selectedLeptons' after disambiguation. Treat with care (= 'False') if running Jetanalyzer more than once
    minLepPt = 10,
    relaxJetId = False,  
    doPuId = True, # Not commissioned in 7.0.X, use the Run1 training for the time being
    recalibrateJets = "MC", # True, False, 'MC', 'Data'
    recalibrationType = "AK4PFchs",
    mcGT     = "MCRUN2_74_V9",
    jecPath = "%s/src/CMGTools/RootTools/data/jec/" % os.environ['CMSSW_BASE'],
    shiftJEC = 0, # set to +1 or -1 to get +/-1 sigma shifts
    smearJets = False,
    shiftJER = 0, # set to +1 or -1 to get +/-1 sigma shifts  
    alwaysCleanPhotons = False,
    cleanJetsFromFirstPhoton = False,
    cleanJetsFromTaus = False,
    cleanJetsFromIsoTracks = False,
    doQG = True,
    cleanGenJetsFromPhoton = False
    )

## Fat Jets Analyzer (generic)
from CMGTools.TTHAnalysis.analyzers.ttHFatJetAnalyzer import ttHFatJetAnalyzer
ttHFatJetAna = cfg.Analyzer(
    ttHFatJetAnalyzer, name = 'ttHFatJetAnalyzer',
    jetCol = 'slimmedJetsAK8',
    jetPt = 100.,
    jetEta = 2.4,
    # v--- not implemented for AK8
    #jetLepDR = 0.4,
    #minLepPt = 10,
    relaxJetId = False,  
    # v--- not implemented for AK8
    #doPuId = False, # Not commissioned in 7.0.X
    #recalibrateJets = False,
    #shiftJEC = 0, # set to +1 or -1 to get +/-1 sigma shifts
    )


# Secondary vertex analyzer
from CMGTools.TTHAnalysis.analyzers.ttHSVAnalyzer import ttHSVAnalyzer
ttHSVAna = cfg.Analyzer(
    ttHSVAnalyzer, name="ttHSVAnalyzer",
)

# Secondary vertex analyzer
from CMGTools.TTHAnalysis.analyzers.ttHHeavyFlavourHadronAnalyzer import ttHHeavyFlavourHadronAnalyzer
ttHHeavyFlavourHadronAna = cfg.Analyzer(
    ttHHeavyFlavourHadronAnalyzer, name="ttHHeavyFlavourHadronAnalyzer",
)


metAna = cfg.Analyzer(
    METAnalyzer, name="metAnalyzer",
    metCollection     = "slimmedMETs",
    noPUMetCollection = "slimmedMETs",
    copyMETsByValue = False,
    doTkMet = False,
    doMetNoPU = True,
    doMetNoMu = True,
    doMetNoEle = False,
    doMetNoPhoton = False,
    recalibrate = False,
    jetAnalyzerCalibrationPostFix = "",
    candidates='packedPFCandidates',
    candidatesTypes='std::vector<pat::PackedCandidate>',
    dzMax = 0.1,
    collectionPostFix = "",
    )

# Core Event Analyzer (computes basic quantities like HT, dilepton masses)
from CMGTools.TTHAnalysis.analyzers.ttHCoreEventAnalyzer import ttHCoreEventAnalyzer
ttHCoreEventAna = cfg.Analyzer(
    ttHCoreEventAnalyzer, name='ttHCoreEventAnalyzer',
    maxLeps = 4, ## leptons to consider
    mhtForBiasedDPhi = "mhtJet40jvec",
    jetForBiasedDPhi = "cleanJets",
    )

# Core sequence of all common modules
dmCoreSequence = [
    skimAnalyzer,
   #eventSelector,
    jsonAna,
    triggerAna,
    pileUpAna,
    genAna,
    genHiggsAna,
    genHFAna,
    pdfwAna,
    vertexAna,
    lepAna,
    photonAna,
    tauAna,
 #   isoTrackAna,
    jetAna,
    metAna,
    ttHCoreEventAna,
    monoJetSkim,
    monoJetCtrlLepSkim,
    triggerFlagsAna,
    eventFlagsAna,
]
