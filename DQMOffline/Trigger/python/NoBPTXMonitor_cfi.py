import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.NoBPTXMonitoring_cfi import NoBPTXMonitoring

hltNoBPTXmonitoring = NoBPTXMonitoring.clone(
    FolderName = 'HLT/EXO/NoBPTX/JetE60/',
    jets = "ak4CaloJets",
    muons = "displacedStandAloneMuons",
    muonSelection = "hitPattern.dtStationsWithValidHits > 3 & hitPattern.numberOfValidMuonRPCHits > 1 & hitPattern.numberOfValidMuonCSCHits < 1",
    jetSelection = "abs(eta) < 1.",

    histoPSet = dict(
        lsPSet = dict(
                nbins = 250,
                xmin  = 0.,
                xmax  = 2500.),
        jetEPSet = dict(
                nbins = 100,
                xmin  = -0.5,
                xmax  = 999.5),

        jetEtaPSet = dict(
                nbins = 100,
                xmin  = -5.,
                xmax  = 5.),

        jetPhiPSet = dict(
                nbins = 64,
                xmin  = -3.2,
                xmax  = 3.2),

        muonPtPSet = dict(
                nbins = 100,
                xmin  = -0.5,
                xmax  = 999.5),
    
        muonEtaPSet = dict(
                nbins = 100,
                xmin  = -5.,
                xmax  = 5.),

        muonPhiPSet = dict(
                nbins = 64,
                xmin  = -3.2,
                xmax  = 3.2),

        bxPSet = dict(
                nbins = 1800)
    ),
 
    numGenericTriggerEventPSet = dict(
        andOr         = False,
        #dbLabel       = "ExoDQMTrigger", # it does not exist yet, we should consider the possibility of using the DB, but as it is now it will need a label per path !                                                                                                           
        andOrHlt      = True,# True:=OR; False:=AND 
        hltInputTag   =  "TriggerResults::HLT",
        hltPaths      = ["HLT_UncorrectedJetE60_NoBPTX3BX_v*"], # HLT_ZeroBias_v*
        #hltDBKey      = "EXO_HLT_NoBPTX", 
        errorReplyHlt =  False,
        verbosityLevel = 1),

    denGenericTriggerEventPSet = dict(
        andOr         =  False,
        dcsInputTag   =  "scalersRawToDigi",
        dcsRecordInputTag = "onlineMetaDataDigis",
        dcsPartitions = [ 24, 25, 26, 27, 28, 29], # 24-27: strip, 28-29: pixel, we should add all other detectors !
        andOrDcs      = False,
        errorReplyDcs = True, 
        verbosityLevel = 1)
)


