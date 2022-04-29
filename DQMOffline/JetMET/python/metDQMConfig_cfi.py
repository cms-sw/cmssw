import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETDQMCleanup_cff import *
from DQMOffline.JetMET.metDiagnosticParameterSet_cfi import *
from DQMOffline.JetMET.metDiagnosticParameterSetMiniAOD_cfi import *

#jet corrector defined in jetMETDQMOfflineSource python file

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
caloMetDQMAnalyzer = DQMEDAnalyzer('METAnalyzer',  
    METType=cms.untracked.string('calo'),
    srcPFlow = cms.InputTag('particleFlow', ''),
    l1algoname = cms.string("L1Tech_BPTX_plus_AND_minus.v0"),
    METCollectionLabel     = cms.InputTag("caloMet"),
    JetCollectionLabel  = cms.InputTag("ak4CaloJets"),
    JetCorrections = cms.InputTag("dqmAk4PFL1FastL2L3ResidualCorrector"),
    muonsrc = cms.InputTag("muons"),

    ptMinCand      = cms.double(1.),
    hcalMin      =cms.double(1.),

    InputJetIDValueMap         = cms.InputTag("ak4JetID"), 
    ptThreshold                =cms.double(30),
    
    FolderName = cms.untracked.string("JetMET/MET/"),

    fillMetHighLevel = cms.bool(True),#fills lumi overview plots

    fillCandidateMaps = cms.bool(False),

    CleaningParameters = cleaningParameters.clone(       
        bypassAllPVChecks = True, #needed for 0T running
        ),
    METDiagonisticsParameters = multPhiCorr_METDiagnostics,

    TriggerResultsLabel  = cms.InputTag("TriggerResults::HLT"),
    FilterResultsLabelMiniAOD  = cms.InputTag("TriggerResults::RECO"),
    FilterResultsLabelMiniAOD2  = cms.InputTag("TriggerResults::reRECO"),

    onlyCleaned                = cms.untracked.bool(True),
    runcosmics                 = cms.untracked.bool(False),  

    LSBegin = cms.int32(0),
    LSEnd   = cms.int32(-1),      

#    HLTPathsJetMB = cms.vstring(),
#    When it is empty, it accepts all the triggers

    triggerSelectedSubFolders = cms.VPSet(
    cms.PSet( label = cms.string('highPtJet'),
        andOr         = cms.bool( False ),    #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_highptjet' ), #overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_PFJet450_v*' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
        stage2 = cms.bool(False),
        l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
        l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
        ReadPrescalesFromFile = cms.bool(False),
    ),
    cms.PSet(label = cms.string('lowPtJet'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_lowptjet' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_PFJet80_v*' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( True ),
        stage2 = cms.bool(False),
        l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
        l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
        ReadPrescalesFromFile = cms.bool(False),
    ),
    cms.PSet(label = cms.string('zeroBias'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
        #hltDBKey       = cms.string( 'jetmet_minbias' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_ZeroBias_v*' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
        stage2 = cms.bool(False),
        l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
        l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
        ReadPrescalesFromFile = cms.bool(False),
    ),
    cms.PSet(label = cms.string('highMET'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_highmet' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_MET250_v*' ), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
        stage2 = cms.bool(False),
        l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
        l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
        ReadPrescalesFromFile = cms.bool(False),
    ),
    #cms.PSet(label = cms.string('singleEle'),
    #    andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
    #    dbLabel        = cms.string("JetMETDQMTrigger"),
    #    hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_ele' ),#overrides hltPaths!
    #    hltPaths       = cms.vstring('HLT_Ele27_WP80_v*' ), 
    #    andOrHlt       = cms.bool( True ),
    #    errorReplyHlt  = cms.bool( False ),
    #),
    cms.PSet(label = cms.string('singleMu'),
        andOr         = cms.bool( False ),     #True -> OR #Comment this line to turn OFF
        dbLabel        = cms.string("JetMETDQMTrigger"),
        hltInputTag    = cms.InputTag( "TriggerResults::HLT" ),
#        hltDBKey       = cms.string( 'jetmet_muon' ),#overrides hltPaths!
        hltPaths       = cms.vstring( 'HLT_IsoMu24_eta2p1_v*', 'HLT_IsoMu27_v*'), 
        andOrHlt       = cms.bool( True ),
        errorReplyHlt  = cms.bool( False ),
        stage2 = cms.bool(False),
        l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
        l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
        ReadPrescalesFromFile = cms.bool(False),
    ) 
    ),
 
    HBHENoiseLabelMiniAOD = cms.string("Flag_HBHENoiseFilter"),
    HBHEIsoNoiseLabelMiniAOD = cms.string("Flag_HBHEIsoNoiseFilter"),

    HcalNoiseRBXCollection     = cms.InputTag("hcalnoise"), 
    HBHENoiseFilterResultLabel = cms.InputTag("HBHENoiseFilterResultProducer", "HBHENoiseFilterResult"),
    HBHENoiseIsoFilterResultLabel = cms.InputTag("HBHENoiseFilterResultProducer", "HBHEIsoNoiseFilterResult"),
    CSCHaloResultLabel = cms.InputTag("CSCTightHaloFilterDQM"), 
    CSCHalo2015ResultLabel = cms.InputTag("CSCTightHalo2015FilterDQM"), 
    EcalDeadCellTriggerPrimitiveFilterLabel = cms.InputTag("EcalDeadCellTriggerPrimitiveFilterDQM"), 
    EcalDeadCellBoundaryEnergyFilterLabel = cms.InputTag("EcalDeadCellBoundaryEnergyFilterDQM"), 
    eeBadScFilterLabel = cms.InputTag("eeBadScFilterDQM"), 
    HcalStripHaloFilterLabel = cms.InputTag("HcalStripHaloFilterDQM"),

    #if changed here, change certification module input in same manner and injetDQMconfig
    pVBin       = cms.int32(100),
    pVMax       = cms.double(100.0),
    pVMin       = cms.double(0.0),

    verbose     = cms.int32(0),

#    etThreshold  = cms.double(2.),

    DCSFilter = cms.PSet(
      DetectorTypes = cms.untracked.string("ecal:hbhe:hf"),
      #DebugOn = cms.untracked.bool(True),
      Filter = cms.untracked.bool(True)
    ),
)

#
# Make changes if using the Stage 2 trigger
#
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(caloMetDQMAnalyzer,
                         triggerSelectedSubFolders = {i: dict(stage2 = True,
                                                              l1tAlgBlkInputTag = "gtStage2Digis",
                                                              l1tExtBlkInputTag = "gtStage2Digis",
                                                              ReadPrescalesFromFile = True) for i in range(0, len(caloMetDQMAnalyzer.triggerSelectedSubFolders))})

pfMetDQMAnalyzer = caloMetDQMAnalyzer.clone(
    METType = 'pf',
    METCollectionLabel     = "pfMet",
    srcPFlow = 'particleFlow',
    JetCollectionLabel  = "ak4PFJets",
    JetCorrections = "dqmAk4PFL1FastL2L3ResidualCorrector",
    CleaningParameters = cleaningParameters.clone(       
        bypassAllPVChecks = False,
        ),
    fillMetHighLevel = False,
    fillCandidateMaps = True,
    # if this flag is changed, the METTypeRECOUncleaned flag in dataCertificationJetMET_cfi.py
    #has to be updated (by a string not pointing to an existing directory)
    onlyCleaned       = False,
    DCSFilter = cms.PSet(
        DetectorTypes = cms.untracked.string("ecal:hbhe:hf:pixel:sistrip:es:muon"),
        #DebugOn = cms.untracked.bool(True),
        Filter = cms.untracked.bool(True)
        ),
)
pfChMetDQMAnalyzer = pfMetDQMAnalyzer.clone(
     METCollectionLabel     = "pfChMet",
     fillCandidateMaps = False,
     onlyCleaned   = True
 )



#both CaloMET and type1 MET only cleaned plots are filled
pfMetT1DQMAnalyzer = caloMetDQMAnalyzer.clone(
    METType = 'pf',
    METCollectionLabel     = "pfMETT1",
    srcPFlow = 'particleFlow',
    JetCollectionLabel  = "ak4PFJetsCHS",
    JetCorrections = "dqmAk4PFCHSL1FastL2L3ResidualCorrector",
    CleaningParameters = cleaningParameters.clone(       
        bypassAllPVChecks = False,
        ),
    fillMetHighLevel = False,
    fillCandidateMaps = False,
    DCSFilter = dict(
        DetectorTypes = "ecal:hbhe:hf:pixel:sistrip:es:muon",
        Filter = True
        ),
)
pfMetDQMAnalyzerMiniAOD = pfMetDQMAnalyzer.clone(
    fillMetHighLevel = True,#fills only lumisec plots
    fillCandidateMaps = False,
    srcPFlow = 'packedPFCandidates',
    METDiagonisticsParameters = multPhiCorr_METDiagnosticsMiniAOD,
    CleaningParameters = cleaningParameters.clone(
        vertexCollection    =  "goodOfflinePrimaryVerticesDQMforMiniAOD",
        ),
    METType = 'miniaod',
    METCollectionLabel  = "slimmedMETs",
    JetCollectionLabel  = "slimmedJets",
    JetCorrections = "" #not called, since corrected by default
)
pfPuppiMetDQMAnalyzerMiniAOD = pfMetDQMAnalyzerMiniAOD.clone(
    fillMetHighLevel = False,#fills only lumisec plots
    fillCandidateMaps = True,
    METType = 'miniaod',
    METCollectionLabel  = "slimmedMETsPuppi",
    JetCollectionLabel  = "slimmedJetsPuppi",
    JetCorrections = "" #not called, since corrected by default
)
