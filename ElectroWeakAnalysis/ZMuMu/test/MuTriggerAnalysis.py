import FWCore.ParameterSet.Config as cms

process = cms.Process("TriggerAnalysis")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

"file:~/www/2010/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133483_331.root" 
    )
)

#import os
#dirname = "/data4/Skimming/SkimResults/133"
#dirlist = os.listdir(dirname)
#basenamelist = os.listdir(dirname + "/")
#for basename in basenamelist:
#                    process.source.fileNames.append("file:" + dirname + "/" + basename)
#                    print "Number of files to process is %s" % (len(process.source.fileNames))


process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*", "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MuTriggerNew.root')
)

# Muon filter, you can choose to add/remove/loose/tighten  cuts (isolation cuts for example)

### muon with all quality cuts except iso 
process.goodMuonsNotIso = cms.EDFilter("MuonSelector",
                                   src = cms.InputTag("muons"),
                                   cut = cms.string('( isGlobalMuon=1 && isTrackerMuon ) && isolationR03().sumPt<1000.0 && abs(innerTrack().dxy)<0.5 && (globalTrack().hitPattern().numberOfValidMuonHits()>0) && (globalTrack.hitPattern().numberOfValidStripHits()>=10) && (globalTrack().normalizedChi2()<10) '), 
                                   filter = cms.bool(True)
                                 )

### all quality cuts
process.goodMuons = cms.EDFilter("MuonSelector",
                                   src = cms.InputTag("muons"),
                                   cut = cms.string('( isGlobalMuon=1 && isTrackerMuon ) && isolationR03().sumPt<3.0 && abs(innerTrack().dxy)<0.5 && (globalTrack().hitPattern().numberOfValidMuonHits()>0) && (globalTrack.hitPattern().numberOfValidStripHits()>=10) && (globalTrack().normalizedChi2()<10) '), 
                                   filter = cms.bool(True)
                                 )

#### no quality cuts
process.goodMuonsNoCuts = cms.EDFilter("MuonSelector",
                                   src = cms.InputTag("muons"),
                                   cut = cms.string('( isGlobalMuon=1 && isTrackerMuon ) '), 
                                   filter = cms.bool(True)
                                 )



process.MuTriggerAnalyzerAllCuts = cms.EDAnalyzer(
    "MuTriggerAnalyzer",
    muons= cms.untracked.InputTag("goodMuons"),
    TrigTag = cms.InputTag("TriggerResults::HLT"),
    triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::HLT" ),  
    hltPath = cms.string("HLT_Mu9"),
##HLT_Mu9
    L3FilterName= cms.string("hltSingleMu9L3Filtered9"),
##hltSingleMu9L3Filtered9
    maxDPtRel = cms.double( 1.0 ),
    maxDeltaR = cms.double( 0.5 ),
    ptMuCut = cms.untracked.double( 5.0 ),
    etaMuCut = cms.untracked.double( 2.1 ),
    ptMax_=cms.double( 40.0 )
)

#import copy
#process.MuTriggerAnalyzerAllCutsButIso=  copy.deepcopy(process.MuTriggerAnalyzerAllCuts)
#process.MuTriggerAnalyzerAllCutsButIso.muons= cms.untracked.InputTag("goodMuonsNotIso") 

#process.MuTriggerAnalyzerNoCuts=  copy.deepcopy(process.MuTriggerAnalyzerAllCuts)
#process.MuTriggerAnalyzerNoCuts.muons= cms.untracked.InputTag("goodMuonsNoCuts")






process.pAllCuts = cms.Path(process.goodMuons* process.MuTriggerAnalyzerAllCuts)
#process.pAllCutsButIso = cms.Path(process.goodMuonsNotIso* process.MuTriggerAnalyzerAllCutsButIso)
##process.pNoCuts = cms.Path(process.goodMuonsNoCuts* process.MuTriggerAnalyzerNoCuts)


