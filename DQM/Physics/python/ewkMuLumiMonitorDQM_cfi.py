import FWCore.ParameterSet.Config as cms


################################################
#####             selection              #######
################################################

CombIsoCuts = cms.PSet(
    IsRelativeIso = cms.untracked.bool(True),
    IsCombinedIso = cms.untracked.bool(True),
    IsoCut03 = cms.untracked.double(0.15),   
#    deltaRTrk = cms.untracked.double(0.3),
    ptThreshold = cms.untracked.double("0.0"), 
 #   deltaRVetoTrk = cms.untracked.double("0.015"), 
    )

TrkIsoCuts = cms.PSet(
    IsRelativeIso = cms.untracked.bool(False),
    IsCombinedIso = cms.untracked.bool(False),
    IsoCut03 = cms.untracked.double(3.0),   
#    deltaRTrk = cms.untracked.double(0.3),
    ptThreshold = cms.untracked.double("0.0"), 
 #   deltaRVetoTrk = cms.untracked.double("0.015"), 
    )



ewkMuLumiMonitorDQM = cms.EDAnalyzer(
    "EwkMuLumiMonitorDQM",
    # isolation cuts
    TrkIsoCuts,
    # for Z/W
    muons= cms.untracked.InputTag("muons"),
    tracks=cms.untracked.InputTag("generalTracks"),
    calotower=cms.untracked.InputTag("towerMaker"),
    metTag=cms.untracked.InputTag("pfMet"),
    METIncludesMuons=cms.untracked.bool(True),
    TrigTag = cms.untracked.InputTag("TriggerResults::HLT"),
    triggerEvent = cms.untracked.InputTag( "hltTriggerSummaryAOD::HLT" ),  
##    hltPath = cms.untracked.string("HLT_Mu11"),
##    L3FilterName= cms.untracked.string("hltSingleMu15L3Filtered15"),
    maxDPtRel = cms.untracked.double( 1.0 ),
    maxDeltaR = cms.untracked.double( 0.2 ),
    ptMuCut = cms.untracked.double( 20.0 ),
    etaMuCut = cms.untracked.double( 2.1 ),
    # W cuts
    mtMin = cms.untracked.double(50.0),
    mtMax = cms.untracked.double(999.),
    acopCut = cms.untracked.double(999.),
    DxyCut = cms.untracked.double(0.2),
)


