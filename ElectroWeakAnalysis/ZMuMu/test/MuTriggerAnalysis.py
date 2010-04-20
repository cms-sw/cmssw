import FWCore.ParameterSet.Config as cms

process = cms.Process("ZMuMuMCanalysis")
process.load("ElectroWeakAnalysis.Skimming.mcTruthForDimuons_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

#  "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132959_1.root",
#  "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132959_2.root",
#  "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132959_3.root",
#  "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132959_4.root",
# "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132959_5.root",
#  "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132959_6.root",
# "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132960.root",
#  "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132961_1.root",
#  "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132961_2.root",
#  "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132961_3.root",
#  "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132961_4.root",
#  "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132965_1.root",
#  "file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132968_1.root",

#    'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132569_michele.root',
# 'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132596_michele.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132597_michele.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132598_1_michele.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132598_2_michele.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132599.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132601_1.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132601_2.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132602_michele.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132605_1.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132605_169509.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132605_2.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132646.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132647.root',
  #'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132648.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132650.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132652.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132653.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132654_1.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132654_2.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132654_3.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132656_1.root',
# 'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132656_2.root',
# 'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132656_3.root',
# 'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132658_1.root',
# 'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132658_2.root',
# 'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132658_3.root',
# 'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132659_1.root',
# 'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132661.root',
# 'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132662.root',
#'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132716_1.root',
# 'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132716_2.root',
# 'file:/data4/Skimming/SkimResults/132/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132716_3.root',
# 'file:/data4/Skimming/SkimResults/132/testEWKMuSkim_L1TG0_4041AllMuAtLeastThreeTracks132440_michele.root',
# 'file:/data4/Skimming/SkimResults/132/testEWKMuSkim_L1TG0_4041AllMuAtLeastThreeTracks132442_michele.root',
    



#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133029_1.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133029_1.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133035_1.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133030_1.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133035_2.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133031_1.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133035_3.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133034_1.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133034_2.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133034_3.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133046_1.root", 
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133046_2.root", 
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133082_1.root", 
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133082_2.root", 
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133082_3.root", 
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133082_4.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133082_5.root", 
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133046_1.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133046_2.root", 
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133082_1.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133082_2.root ",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133082_3.root", 
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133082_4.root ",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133082_5.root",
#"file:/data4/Skimming/SkimResults/133/EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks133483_2.root" 
    )
)

import os
dirname = "/data4/Skimming/SkimResults/133"
dirlist = os.listdir(dirname)
basenamelist = os.listdir(dirname + "/")
for basename in basenamelist:
                    process.source.fileNames.append("file:" + dirname + "/" + basename)
                    print "Number of files to process is %s" % (len(process.source.fileNames))

#dirname = "/data4/Skimming/SkimResults/132"
#dirlist = os.listdir(dirname)
#basenamelist = os.listdir(dirname + "/")
#for basename in basenamelist:
#                    if basename != 'testEWKMuSkim_L1TG0_132440_1.root' and basename!='EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132569_michele.root' and basename!= 'EWKMuSkim_L1TG04041_AllMuAtLeastThreeTracks132596_michele.root':
#                        process.source.fileNames.append("file:" + dirname + "/" + basename)
#                        print "Number of files to process is %s" % (len(process.source.fileNames))                    
                                    
                                    

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*", "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MuTriggerStudy_133_AllqualityCutsButIso.root')
)

# Muon filter
process.goodMuons = cms.EDFilter("MuonSelector",
                                   src = cms.InputTag("muons"),
                                   cut = cms.string('( isGlobalMuon=1 && isTrackerMuon ) && isolationR03().sumPt<1000.0 && abs(innerTrack().dxy)<0.5 && (globalTrack().hitPattern().numberOfValidMuonHits()>0) && (globalTrack.hitPattern().numberOfValidStripHits()>=10) && (globalTrack().normalizedChi2()<10) '), 
                                   filter = cms.bool(True)
                                 )



process.MuTriggerAnalyzer = cms.EDAnalyzer(
    "MuTriggerAnalyzer",
    muons= cms.untracked.InputTag("goodMuons"),
    TrigTag = cms.InputTag("TriggerResults::HLT"),
    triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::HLT" ),  
    hltPath = cms.string("HLT_L2Mu9"),
##HLT_L2Mu9
    L3FilterName= cms.string("hltL2Mu9L2Filtered9"),
##hltL2Mu9L2Filtered9
    maxDPtRel = cms.double( 1.0 ),
    maxDeltaR = cms.double( 0.5 ),
    ptMuCut = cms.untracked.double( 5.0 ),
    etaMuCut = cms.untracked.double( 2.1 ),
    ptMax_=cms.double( 40.0 )
)







process.p = cms.Path(process.goodMuons* process.MuTriggerAnalyzer)


