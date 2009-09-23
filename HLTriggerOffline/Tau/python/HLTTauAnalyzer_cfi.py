import FWCore.ParameterSet.Config as cms

hltTauAnalyzer = cms.EDAnalyzer("HLTTauAnalyzer",
    mcDeltaRLep = cms.double(0.5),
    #  Max delta R between generator and L1/HLT candidates
    mcDeltaRTau = cms.double(0.5),
    #  Output ROOT file and text file
    rootFile = cms.untracked.string('HLTAnalyzer.root'),
    logFile = cms.untracked.string('HLTAnalyzer.log'),
    # Level 2 MET
    HLTMETFilter = cms.InputTag("hlt1METSingleTau"),
    l2TauJets = cms.VInputTag(),
    l2TauJetsFiltered = cms.VInputTag(),
    l25TauJets = cms.VInputTag(),
    l3TauJets = cms.VInputTag(),
    #  Switch for signal/background datasets
    isSignal = cms.bool(True),
    #  Products along the HLT paths
    hltLeptonSrc = cms.VInputTag(),
    UsingMET = cms.bool(False),
    #  Level 1 trigger bit: "L1_IsoEG10_TauJet20" "L1_Mu5_TauJet20" "L1_SingleTauJet80" "L1_DoubleTauJet40" "L1_TauJet30_ETIM30"
    l1TauTrigger = cms.string(''),
    debug = cms.int32(1000),
    # Number of tau candidates to be checked at every step of the trigger path 
    # 1 = singleTau, singleTauMET, lepton+tau, 2 = double tau
    nbLeps = cms.int32(0),
    #  Monte-Carlo products from HLTTauMcInfo producer
    mcProducts = cms.VInputTag(cms.InputTag("TauMcInfoProducer","Jets"), cms.InputTag("TauMcInfoProducer","Leptons"), cms.InputTag("TauMcInfoProducer","Neutrina")),
    nbTaus = cms.int32(1)
)


