import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.Heppy.analyzers.core.all import *
from PhysicsTools.Heppy.analyzers.objects.all import *
from PhysicsTools.Heppy.analyzers.gen.all import *
from CMGTools.HToZZ4L.analyzers.FourLeptonAnalyzer import *
from CMGTools.HToZZ4L.analyzers.FourLeptonAnalyzer2P2F import *
from CMGTools.HToZZ4L.analyzers.FourLeptonAnalyzerRelaxIdIso import *
from CMGTools.HToZZ4L.analyzers.FourLeptonAnalyzer3P1F import *
from CMGTools.HToZZ4L.analyzers.FourLeptonAnalyzerSS import *
from CMGTools.HToZZ4L.analyzers.FourLeptonEventSkimmer import *

from CMGTools.HToZZ4L.analyzers.FSRPhotonMaker import *
from CMGTools.HToZZ4L.analyzers.GenFSRAnalyzer import *
from CMGTools.HToZZ4L.analyzers.fourLeptonTree import *
from CMGTools.HToZZ4L.samples.samples_13TeV_Spring15 import triggers_mumu, triggers_ee, triggers_mue, triggers_3e, triggers_3mu, triggers_2mu1e, triggers_2e1mu, triggers_mumu_run1, triggers_ee_run1, triggers_mue_run1, triggers_1mu_iso
from CMGTools.RootTools.samples.triggers_8TeV import triggers_1mu_8TeV, triggers_mumu_8TeV, triggers_mue_8TeV, triggers_ee_8TeV

import os


PDFWeights = []

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


genFSRAna = cfg.Analyzer(
    GenFSRAnalyzer, name="GenFSRAnalyzer"
    )


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
        # Doubles
        'DoubleMu' : triggers_mumu,
        'DoubleEl' : triggers_ee,
        'MuEG'     : triggers_mue,
        # Triples
        'TripleEl' : triggers_3e,
        'TripleMu' : triggers_3mu,
        'DoubleMuEl' : triggers_2mu1e,
        'DoubleElMu' : triggers_2e1mu,
        # Singles
        'SingleMu' : triggers_1mu_iso,
        # 8 TeV (and closest equivalent in spring 15 mc)
        'SingleMu_8TeV' : triggers_1mu_8TeV + triggers_1mu_iso,
        'DoubleMu_8TeV' : triggers_mumu_8TeV + triggers_mumu_run1,
        'MuEG_8TeV'     : triggers_mue_8TeV + triggers_mue_run1,
        'DoubleEl_8TeV' : triggers_ee_8TeV + triggers_ee_run1,
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

pdfwAna = cfg.Analyzer(
    PDFWeightsAnalyzer, name="PDFWeightsAnalyzer",
    PDFWeights = [ pdf for pdf,num in PDFWeights ]
    )



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
    doSegmentBasedMuonCleaning=True,
    notCleaningElectrons=True, # no deltaR(ele,mu) cleaning at this step
    # inclusive very loose muon selection
    inclusive_muon_id  = "POG_Global_OR_TMArbitrated",
    inclusive_muon_pt  = 5,
    inclusive_muon_eta = 2.4,
    inclusive_muon_dxy = 0.5,
    inclusive_muon_dz  = 1.0,
    muon_dxydz_track   = "muonBestTrack",
    # loose muon selection
    loose_muon_id     = "POG_Global_OR_TMArbitrated",
    loose_muon_pt     = 5,
    loose_muon_eta    = 2.4,
    loose_muon_dxy    = 0.5,
    loose_muon_dz     = 1,
    loose_muon_isoCut = lambda muon : muon.sip3D() < 4 and muon.muonBestTrackType() != 2,
    # inclusive very loose electron selection
    inclusive_electron_id  = "",
    inclusive_electron_pt  = 7,
    inclusive_electron_eta = 2.5,
    inclusive_electron_dxy = 0.5,
    inclusive_electron_dz  = 1.0,
    inclusive_electron_lostHits = 1.0,
    # loose electron selection
    loose_electron_id     = "",
    loose_electron_pt     = 7,
    loose_electron_eta    = 2.5,
    loose_electron_dxy    = 0.5,
    loose_electron_dz     = 1.0,
    loose_electron_isoCut = lambda x: x.sip3D() < 4,
    loose_electron_lostHits = 1.0,
    # muon isolation correction method (can be "rhoArea" or "deltaBeta")
    mu_isoCorr = "deltaBeta" ,
    mu_effectiveAreas = "Phys14_25ns_v1", #(can be 'Data2012' or 'Phys14_25ns_v1')
    mu_tightId = "POG_ID_Loose",
    # electron isolation correction method (can be "rhoArea" or "deltaBeta")
    ele_isoCorr = "rhoArea" ,
    el_effectiveAreas = "Phys14_25ns_v1" , #(can be 'Data2012' or 'Phys14_25ns_v1')
    ele_tightId = "MVA_ID_NonTrig_Phys14Fix_HZZ",
    # Mini-isolation, with pT dependent cone: will fill in the miniRelIso, miniRelIsoCharged, miniRelIsoNeutral variables of the leptons (see https://indico.cern.ch/event/368826/ )
    doMiniIsolation = False, # off by default since it requires access to all PFCandidates 
    packedCandidates = 'packedPFCandidates',
    miniIsolationPUCorr = 'rhoArea', # Allowed options: 'rhoArea' (EAs for 03 cone scaled by R^2), 'deltaBeta', 'raw' (uncorrected), 'weights' (delta beta weights; not validated)
    miniIsolationVetoLeptons = None, # use 'inclusive' to veto inclusive leptons and their footprint in all isolation cones
    # minimum deltaR between a loose electron and a loose muon (on overlaps, discard the electron)
    min_dr_electron_muon = 100.0,
    # do MC matching 
    do_mc_match = True, # note: it will in any case try it only on MC, not on data
    match_inclusiveLeptons = False, # match to all inclusive leptons
    )

from CMGTools.HToZZ4L.analyzers.ElectronMuonCleaner import ElectronMuonCleaner
eleMuClean = cfg.Analyzer(
    ElectronMuonCleaner, name='eleMuClean',
    selectedMuCut = lambda mu : mu.tightId(), #isPFMuon() or mu.isGlobalMuon(),
    otherMuCut    = lambda mu : False, # (mu.isPFMuon() or mu.isGlobalMuon()) and muon.muonBestTrackType() != 2, # uncomment to include also muons with sip > 4
    mustClean = lambda ele, mu, dr: dr < 0.05
)

## Jets Analyzer (generic)
jetAna = cfg.Analyzer(
    JetAnalyzer, name='jetAnalyzer',
    jetCol = 'slimmedJets',
    copyJetsByValue = False,      #Whether or not to copy the input jets or to work with references (should be 'True' if JetAnalyzer is run more than once)
    genJetCol = 'slimmedGenJets',
    rho = ('fixedGridRhoFastjetAll','',''),
    jetPt = 30.,
    jetEta = 4.7,
    jetEtaCentral = 4.7,
    jetLepDR = 0.4,
    jetLepArbitration = (lambda jet,lepton : lepton), # you can decide which to keep in case of overlaps; e.g. if the jet is b-tagged you might want to keep the jet
    cleanSelectedLeptons = True, #Whether to clean 'selectedLeptons' after disambiguation. Treat with care (= 'False') if running Jetanalyzer more than once
    minLepPt = 0,
    lepSelCut = lambda lepton : lepton.tightId() and lepton.relIso04 < (0.4 if abs(lepton.pdgId())==13 else 0.5),
    relaxJetId = False,  
    doPuId = True,
    recalibrateJets = False, # True, False, 'MC', 'Data'
    recalibrationType = "AK4PFchs",
    mcGT     = "Summer15_V5_p6_MC",
    jecPath = "${CMSSW_BASE}/src/CMGTools/RootTools/data/jec/",
    shiftJEC = 0, # set to +1 or -1 to get +/-1 sigma shifts
    addJECShifts = False,
    smearJets = False,
    shiftJER = 0, # set to +1 or -1 to get +/-1 sigma shifts  
    alwaysCleanPhotons = False,
    cleanJetsFromFirstPhoton = False,
    cleanJetsFromTaus = False,
    cleanJetsFromIsoTracks = False,
    doQG = False,
    cleanGenJetsFromPhoton = False,
    )


metAna = cfg.Analyzer(
    METAnalyzer, name="metAnalyzer",
    metCollection     = "slimmedMETs",
    noPUMetCollection = "slimmedMETs",    
    copyMETsByValue = False,
    doTkMet = False,
    doMetNoPU = False,
    doMetNoMu = False,
    doMetNoEle = False,
    doMetNoPhoton = False,
    recalibrate = False,
    jetAnalyzerCalibrationPostFix = "",
    candidates='packedPFCandidates',
    candidatesTypes='std::vector<pat::PackedCandidate>',
    dzMax = 0.1,
    collectionPostFix = "",
    )



fsrPhotonMaker = cfg.Analyzer(
    FSRPhotonMaker, name="fsrPhotonMaker",
    leptons="selectedLeptons",
    electronID = lambda x: True, #x.electronID("POG_MVA_ID_Run2_NonTrig_HZZ")
    electronVeto = "superclusterEta", # alternatives: "electronEta" and in the future "pfCandReference"
)


fourLeptonAnalyzerSignal = cfg.Analyzer(
    FourLeptonAnalyzer, name="fourLeptonAnalyzerSignal",
    tag = "Signal",
    attachFsrToGlobalClosestLeptonOnly = True
)

fourLeptonAnalyzer2P2F = cfg.Analyzer(
    FourLeptonAnalyzer2P2F, name="fourLeptonAnalyzer2P2F",
    tag = "2P2F",
    attachFsrToGlobalClosestLeptonOnly = True
)

fourLeptonAnalyzer3P1F = cfg.Analyzer(
    FourLeptonAnalyzer3P1F, name="fourLeptonAnalyzer3P1F",
    tag = "3P1F",
    attachFsrToGlobalClosestLeptonOnly = True
)

fourLeptonAnalyzerSS = cfg.Analyzer(
    FourLeptonAnalyzerSS, name="fourLeptonAnalyzerSS",
    tag = "SS",
    attachFsrToGlobalClosestLeptonOnly = True
)

fourLeptonAnalyzerRelaxIdIso = cfg.Analyzer(
    FourLeptonAnalyzerRelaxIdIso, name="fourLeptonAnalyzerRelaxIdIso",
    tag = "RelaxIdIso",
    maxCand = 999, # save all, not just the best one
    attachFsrToGlobalClosestLeptonOnly = True
)

fourLeptonEventSkimmer = cfg.Analyzer(
    FourLeptonEventSkimmer, name="fourLeptonEventSkimmer",
    required = ['bestFourLeptonsSignal','bestFourLeptons2P2F','bestFourLeptons3P1F','bestFourLeptonsSS', 'bestFourLeptonsRelaxIdIso' ]

)

treeProducer = cfg.Analyzer(
     AutoFillTreeProducer, name='fourLeptonTreeProducer',
     vectorTree = False,
     saveTLorentzVectors = False,  # can set to True to get also the TLorentzVectors, but trees will be bigger
     globalVariables = hzz_globalVariables,
     globalObjects = hzz_globalObjects,
     collections = hzz_collections,
     defaultFloatType = 'F',
)




# Core sequence of all common modules
hzz4lCoreSequence = [
    skimAnalyzer,
    genAna,
#    genFSRAna,
   #eventSelector,
    jsonAna,
    triggerAna,
    pileUpAna,
    vertexAna,
    lepAna,
    eleMuClean,
    jetAna,
    metAna,
    triggerFlagsAna,
    fsrPhotonMaker,
    fourLeptonAnalyzerSignal, 
    fourLeptonAnalyzer2P2F,
    fourLeptonAnalyzer3P1F,
    fourLeptonAnalyzerSS,
    fourLeptonAnalyzerRelaxIdIso,
    fourLeptonEventSkimmer,
    treeProducer
]
