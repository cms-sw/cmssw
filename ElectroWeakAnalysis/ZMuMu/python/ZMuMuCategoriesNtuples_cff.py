import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *
import copy


#### ntuple for loose cuts

goodZToMuMuEdmNtupleLoose = cms.EDProducer(
    "ZToLLEdmNtupleDumper",
    zBlocks = cms.VPSet(
    cms.PSet(
        zName = cms.string("zGolden"),
        z = cms.InputTag("goodZToMuMuAtLeast1HLTLoose"),
        zGenParticlesMatch = cms.InputTag(""),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        primaryVertices = cms.InputTag("offlinePrimaryVerticesWithBS"), 
        ptThreshold = cms.double(1.5),
        etEcalThreshold = cms.double(0.2),
        etHcalThreshold = cms.double(0.5),
        deltaRVetoTrk = cms.double(0.015),
        deltaRTrk = cms.double(0.3),
        deltaREcal = cms.double(0.25),
        deltaRHcal = cms.double(0.25),
        alpha = cms.double(0.),
        beta = cms.double(-0.75),
        relativeIsolation = cms.bool(False),
        hltPath = cms.string("HLT_Mu9")
      ),
     )
)

goodZToMuMuPathLoose.__iadd__(goodZToMuMuEdmNtupleLoose)
goodZToMuMuPathLoose.setLabel("goodZToMuMuLoose")


## goodZToMuMu2HLTEdmNtupleLoose = copy.deepcopy(goodZToMuMuEdmNtupleLoose)
## goodZToMuMu2HLTEdmNtupleLoose.zBlocks[0].z = cms.InputTag("goodZToMuMu2HLTLoose")
## goodZToMuMu2HLTEdmNtupleLoose.zBlocks[0].zName = cms.string("zGolden2HLT")
## goodZToMuMu2HLTPathLoose.__iadd__(goodZToMuMu2HLTEdmNtupleLoose)
## goodZToMuMu2HLTPathLoose.setLabel("goodZToMuMu2HLTLoose")


## goodZToMuMu1HLTEdmNtupleLoose = copy.deepcopy(goodZToMuMuEdmNtupleLoose)
## goodZToMuMu1HLTEdmNtupleLoose.zBlocks[0].z = cms.InputTag("goodZToMuMu1HLTLoose")
## goodZToMuMu1HLTEdmNtupleLoose.zBlocks[0].zName = cms.string("zGolden1HLT")
## goodZToMuMu1HLTPathLoose.__iadd__(goodZToMuMu1HLTEdmNtupleLoose)
## goodZToMuMu1HLTPathLoose.setLabel("goodZToMuMu1HLTLoose")

## oneNonIsolatedZToMuMuEdmNtuple = copy.deepcopy(goodZToMuMuEdmNtuple)
## oneNonIsolatedZToMuMuEdmNtuple.zBlocks[0].z = cms.InputTag("oneNonIsolatedZToMuMuAtLeast1HLT")
## oneNonIsolatedZToMuMuEdmNtuple.zBlocks[0].zName = cms.string("z1NotIso")
## oneNonIsolatedZToMuMuPath.__iadd__(oneNonIsolatedZToMuMuEdmNtuple)
## oneNonIsolatedZToMuMuPath.setLabel("oneNonIsolatedZToMuMu")

## twoNonIsolatedZToMuMuEdmNtuple = copy.deepcopy(goodZToMuMuEdmNtuple)
## twoNonIsolatedZToMuMuEdmNtuple.zBlocks[0].z = cms.InputTag("twoNonIsolatedZToMuMuAtLeast1HLT")
## twoNonIsolatedZToMuMuEdmNtuple.zBlocks[0].zName = cms.string("z2NotIso")
## twoNonIsolatedZToMuMuPath.__iadd__(twoNonIsolatedZToMuMuEdmNtuple)
## twoNonIsolatedZToMuMuPath.setLabel("twoNonIsolatedZToMuMu")

## goodZToMuMuSameCharge2HLTEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtupleLoose)
## goodZToMuMuSameCharge2HLTEdmNtupleLoose.zBlocks[0].z = cms.InputTag("goodZToMuMuSameCharge2HLTLoose")
## goodZToMuMuSameCharge2HLTEdmNtupleLoose.zBlocks[0].zName = cms.string("zSameCharge2HLT")
## goodZToMuMuSameCharge2HLTPathLoose.__iadd__(goodZToMuMuSameCharge2HLTEdmNtupleLoose)
## goodZToMuMuSameCharge2HLTPathLoose.setLabel("goodZToMuMuSameCharge2HLTLoose")


## goodZToMuMuSameCharge1HLTEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtupleLoose)
## goodZToMuMuSameCharge1HLTEdmNtupleLoose.zBlocks[0].z = cms.InputTag("goodZToMuMuSameCharge1HLTLoose")
## goodZToMuMuSameCharge1HLTEdmNtupleLoose.zBlocks[0].zName = cms.string("zSameCharge1HLT")
## goodZToMuMuSameCharge1HLTPathLoose.__iadd__(goodZToMuMuSameCharge1HLTEdmNtupleLoose)
## goodZToMuMuSameCharge1HLTPathLoose.setLabel("goodZToMuMuSameCharge1HLTLoose")

goodZToMuMuAB1HLTEdmNtupleLoose = copy.deepcopy(goodZToMuMuEdmNtupleLoose)
goodZToMuMuAB1HLTEdmNtupleLoose.zBlocks[0].z = cms.InputTag("goodZToMuMuAB1HLTLoose")
goodZToMuMuAB1HLTEdmNtupleLoose.zBlocks[0].zName = cms.string("zGoldenAB1HLT")
goodZToMuMuAB1HLTPathLoose.__iadd__(goodZToMuMuAB1HLTEdmNtupleLoose)
goodZToMuMuAB1HLTPathLoose.setLabel("goodZToMuMuAB1HLTLoose")

goodZToMuMuBB2HLTEdmNtupleLoose = copy.deepcopy(goodZToMuMuEdmNtupleLoose)
goodZToMuMuBB2HLTEdmNtupleLoose.zBlocks[0].z = cms.InputTag("goodZToMuMuBB2HLTLoose")
goodZToMuMuBB2HLTEdmNtupleLoose.zBlocks[0].zName = cms.string("zGoldenBB2HLT")
goodZToMuMuBB2HLTPathLoose.__iadd__(goodZToMuMuBB2HLTEdmNtupleLoose)
goodZToMuMuBB2HLTPathLoose.setLabel("goodZToMuMuBB2HLTLoose")



goodZToMuMuSameChargeEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtupleLoose)
goodZToMuMuSameChargeEdmNtupleLoose.zBlocks[0].z = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLTLoose")
goodZToMuMuSameChargeEdmNtupleLoose.zBlocks[0].zName = cms.string("zSameCharge")
goodZToMuMuSameChargePathLoose.__iadd__(goodZToMuMuSameChargeEdmNtupleLoose)
goodZToMuMuSameChargePathLoose.setLabel("goodZToMuMuSameChargeLoose")



goodZToMuMuOneStandAloneEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtupleLoose)
goodZToMuMuOneStandAloneEdmNtupleLoose.zBlocks[0].z=cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLTLoose")
goodZToMuMuOneStandAloneEdmNtupleLoose.zBlocks[0].zName=cms.string("zMuSta")
goodZToMuMuOneStandAloneMuonPathLoose.__iadd__(goodZToMuMuOneStandAloneEdmNtupleLoose)
goodZToMuMuOneStandAloneMuonPathLoose.setLabel("goodZToMuMuOneStandAloneMuonLoose")

goodZToMuMuOneTrackEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtupleLoose)
goodZToMuMuOneTrackEdmNtupleLoose.zBlocks[0].z=cms.InputTag("goodZToMuMuOneTrackFirstHLTLoose")
goodZToMuMuOneTrackEdmNtupleLoose.zBlocks[0].zName=cms.string("zMuTrk")
goodZToMuMuOneTrackPathLoose.__iadd__(goodZToMuMuOneTrackEdmNtupleLoose)
goodZToMuMuOneTrackPathLoose.setLabel("goodZToMuMuOneTrackLoose")


goodZToMuMuOneTrackerMuonEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtupleLoose)
goodZToMuMuOneTrackerMuonEdmNtupleLoose.zBlocks[0].z=cms.InputTag("goodZToMuMuOneTrackerMuonFirstHLTLoose")
goodZToMuMuOneTrackerMuonEdmNtupleLoose.zBlocks[0].zName=cms.string("zMuTrkMu")
goodZToMuMuOneTrackerMuonPathLoose.__iadd__(goodZToMuMuOneTrackerMuonEdmNtupleLoose)
goodZToMuMuOneTrackerMuonPathLoose.setLabel("goodZToMuMuOneTrackerMuonLoose")


ntuplesOut = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('NtupleLoose_test.root'),
    outputCommands = cms.untracked.vstring(
      "drop *",
#      "keep *_goodZToMuMuOneStandAloneMuonNtuple_*_*",
      "keep *_goodZToMuMuEdmNtupleLoose_*_*",
  #    "keep *_goodZToMuMu1HLTEdmNtupleLoose_*_*",
  #    "keep *_goodZToMuMu2HLTEdmNtupleLoose_*_*",
      "keep *_goodZToMuMuAB1HLTEdmNtupleLoose_*_*",
      "keep *_goodZToMuMuBB2HLTEdmNtupleLoose_*_*",
      "keep *_goodZToMuMuSameChargeEdmNtupleLoose_*_*",
#      "keep *_goodZToMuMuSameCharge1HLTEdmNtupleLoose_*_*",
   #   "keep *_nonIsolatedZToMuMuEdmNtuple_*_*",
  #    "keep *_oneNonIsolatedZToMuMuEdmNtuple_*_*",
  #    "keep *_twoNonIsolatedZToMuMuEdmNtuple_*_*",
      "keep *_goodZToMuMuOneStandAloneEdmNtupleLoose_*_*",
      "keep *_goodZToMuMuOneTrackEdmNtupleLoose_*_*",
      "keep *_goodZToMuMuOneTrackerMuonEdmNtupleLoose_*_*",
 #     "keep *_goodZToMuMu2HLTVtxedNtuple_*_*",
      
    ),
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring(
        "goodZToMuMuPathLoose",
   #     "goodZToMuMu1HLTPathLoose",
   #     "goodZToMuMu2HLTPathLoose",
         "goodZToMuMuAB1HLTPathLoose",
         "goodZToMuMuBB2HLTPathLoose",
        "goodZToMuMuSameChargePathLoose",
#        "goodZToMuMuSameCharge1HLTPathLoose",
   #     "nonIsolatedZToMuMuPath",
   #     "oneNonIsolatedZToMuMuPath",
   #     "twoNonIsolatedZToMuMuPath",
        "goodZToMuMuOneTrackPathLoose",
        "goodZToMuMuOneTrackerMuonPathLoose",
        "goodZToMuMuOneStandAloneMuonPathLoose",
      )
    )
)

ntuplesOut.setLabel("ntuplesOut")
NtuplesOut.__iadd__(ntuplesOut)
NtuplesOut.setLabel("NtuplesOut")


## #### ntuple for tight cuts

## goodZToMuMuEdmNtupleTight = cms.EDProducer(
##     "ZToLLEdmNtupleDumper",
##     zBlocks = cms.VPSet(
##     cms.PSet(
##         zName = cms.string("zGoldenTight"),
##         z = cms.InputTag("goodZToMuMuAtLeast1HLTTight"),
##         zGenParticlesMatch = cms.InputTag(""),
##         beamSpot = cms.InputTag("offlineBeamSpot"),
##         primaryVertices = cms.InputTag("offlinePrimaryVerticesWithBS"), 
##         ptThreshold = cms.double(1.5),
##         etEcalThreshold = cms.double(0.2),
##         etHcalThreshold = cms.double(0.5),
##         deltaRVetoTrk = cms.double(0.015),
##         deltaRTrk = cms.double(0.3),
##         deltaREcal = cms.double(0.25),
##         deltaRHcal = cms.double(0.25),
##         alpha = cms.double(0.),
##         beta = cms.double(-0.75),
##         relativeIsolation = cms.bool(False)
##       ),
##      )
## )



## goodZToMuMu2HLTEdmNtupleTight = copy.deepcopy(goodZToMuMuEdmNtupleTight)
## goodZToMuMu2HLTEdmNtupleTight.zBlocks[0].z = cms.InputTag("goodZToMuMu2HLTTight")
## goodZToMuMu2HLTEdmNtupleTight.zBlocks[0].zName = cms.string("zGolden2HLTTight")
## goodZToMuMu2HLTPathTight.__iadd__(goodZToMuMu2HLTEdmNtupleTight)
## goodZToMuMu2HLTPathTight.setLabel("goodZToMuMu2HLTTight")


## goodZToMuMu1HLTEdmNtupleTight = copy.deepcopy(goodZToMuMuEdmNtupleTight)
## goodZToMuMu1HLTEdmNtupleTight.zBlocks[0].z = cms.InputTag("goodZToMuMu1HLTTight")
## goodZToMuMu1HLTEdmNtupleTight.zBlocks[0].zName = cms.string("zGolden1HLTTight")
## goodZToMuMu1HLTPathTight.__iadd__(goodZToMuMu1HLTEdmNtupleTight)
## goodZToMuMu1HLTPathTight.setLabel("goodZToMuMu1HLTTight")

## oneNonIsolatedZToMuMuEdmNtupleTight = copy.deepcopy(goodZToMuMuEdmNtupleTight)
## oneNonIsolatedZToMuMuEdmNtupleTight.zBlocks[0].z = cms.InputTag("oneNonIsolatedZToMuMuAtLeast1HLTTight")
## oneNonIsolatedZToMuMuEdmNtupleTight.zBlocks[0].zName = cms.string("z1NotIsoTight")
## oneNonIsolatedZToMuMuPathTight.__iadd__(oneNonIsolatedZToMuMuEdmNtupleTight)
## oneNonIsolatedZToMuMuPathTight.setLabel("oneNonIsolatedZToMuMuTight")

## twoNonIsolatedZToMuMuEdmNtupleTight = copy.deepcopy(goodZToMuMuEdmNtupleTight)
## twoNonIsolatedZToMuMuEdmNtupleTight.zBlocks[0].z = cms.InputTag("twoNonIsolatedZToMuMuAtLeast1HLTTight")
## twoNonIsolatedZToMuMuEdmNtupleTight.zBlocks[0].zName = cms.string("z2NotIsoTight")
## twoNonIsolatedZToMuMuPathTight.__iadd__(twoNonIsolatedZToMuMuEdmNtupleTight)
## twoNonIsolatedZToMuMuPathTight.setLabel("twoNonIsolatedZToMuMuTight")

## goodZToMuMuSameCharge2HLTEdmNtupleTight= copy.deepcopy(goodZToMuMuEdmNtupleTight)
## goodZToMuMuSameCharge2HLTEdmNtupleTight.zBlocks[0].z = cms.InputTag("goodZToMuMuSameCharge2HLTTight")
## goodZToMuMuSameCharge2HLTEdmNtupleTight.zBlocks[0].zName = cms.string("zSameCharge2HLTTight")
## goodZToMuMuSameCharge2HLTPathTight.__iadd__(goodZToMuMuSameCharge2HLTEdmNtupleTight)
## goodZToMuMuSameCharge2HLTPathTight.setLabel("goodZToMuMuSameCharge2HLTTight")


## goodZToMuMuSameCharge1HLTEdmNtupleTight= copy.deepcopy(goodZToMuMuEdmNtupleTight)
## goodZToMuMuSameCharge1HLTEdmNtupleTight.zBlocks[0].z = cms.InputTag("goodZToMuMuSameCharge1HLTTight")
## goodZToMuMuSameCharge1HLTEdmNtupleTight.zBlocks[0].zName = cms.string("zSameCharge1HLTTight")
## goodZToMuMuSameCharge1HLTPathTight.__iadd__(goodZToMuMuSameCharge1HLTEdmNtupleTight)
## goodZToMuMuSameCharge1HLTPathTight.setLabel("goodZToMuMuSameCharge1HLTTight")


## goodZToMuMuOneStandAloneEdmNtupleTight= copy.deepcopy(goodZToMuMuEdmNtupleTight)
## goodZToMuMuOneStandAloneEdmNtupleTight.zBlocks[0].z=cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLTTight")
## goodZToMuMuOneStandAloneEdmNtupleTight.zBlocks[0].zName=cms.string("zMuStaTight")
## goodZToMuMuOneStandAloneMuonPathTight.__iadd__(goodZToMuMuOneStandAloneEdmNtupleTight)
## goodZToMuMuOneStandAloneMuonPathTight.setLabel("goodZToMuMuOneStandAloneMuonTight")

## goodZToMuMuOneTrackEdmNtupleTight= copy.deepcopy(goodZToMuMuEdmNtupleTight)
## goodZToMuMuOneTrackEdmNtupleTight.zBlocks[0].z=cms.InputTag("goodZToMuMuOneTrackFirstHLTTight")
## goodZToMuMuOneTrackEdmNtupleTight.zBlocks[0].zName=cms.string("zMuTrkTight")
## goodZToMuMuOneTrackPathTight.__iadd__(goodZToMuMuOneTrackEdmNtupleTight)
## goodZToMuMuOneTrackPathTight.setLabel("goodZToMuMuOneTrackTight")

## ntuplesOutTight = cms.OutputModule(
##     "PoolOutputModule",
##     fileName = cms.untracked.string('NtupleTight_test.root'),
##     outputCommands = cms.untracked.vstring(
##       "drop *",
## #      "keep *_goodZToMuMuOneStandAloneMuonNtuple_*_*",
##       "keep *_goodZToMuMuEdmNtupleTight_*_*",
##       "keep *_goodZToMuMu1HLTEdmNtupleTight_*_*",
##       "keep *_goodZToMuMu2HLTEdmNtupleTight_*_*",
##       "keep *_goodZToMuMuSameCharge2HLTEdmNtupleTight_*_*",
##       "keep *_goodZToMuMuSameCharge1HLTEdmNtupleTight_*_*",
##       "keep *_nonIsolatedZToMuMuEdmNtupleTight_*_*",
##       "keep *_oneNonIsolatedZToMuMuEdmNtupleTight_*_*",
##       "keep *_twoNonIsolatedZToMuMuEdmNtupleTight_*_*",
##       "keep *_goodZToMuMuOneStandAloneEdmNtupleTight_*_*",
##       "keep *_goodZToMuMuOneTrackEdmNtupleTight_*_*",
##  #     "keep *_goodZToMuMu2HLTVtxedNtuple_*_*",
      
##     ),
##     SelectEvents = cms.untracked.PSet(
##       SelectEvents = cms.vstring(
##         "goodZToMuMuPathTight",
##         "goodZToMuMu1HLTPathTight",
##         "goodZToMuMu2HLTPathTight",
##         "goodZToMuMuSameCharge2HLTPathTight",
##         "goodZToMuMuSameCharge1HLTPathTight",
##         "nonIsolatedZToMuMuPathTight",
##         "oneNonIsolatedZToMuMuPathTight",
##         "twoNonIsolatedZToMuMuPathTight",
##         "goodZToMuMuOneTrackPathTight",
##         "goodZToMuMuOneStandAloneMuonPathTight",
##       )
##     )
## )
  



## ntuplesOutTight.setLabel("ntuplesOutTight")
## NtuplesOutTight.__iadd__(ntuplesOutTight)
## NtuplesOutTight.setLabel("NtuplesOutTight")
