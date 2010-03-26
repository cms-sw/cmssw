import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *
import copy

#### vertex refit for loose cut

goodZToMuMuVtxedAtLeast1HLTLoose = cms.EDProducer(
    "KalmanVertexFitCompositeCandProducer",
    src = cms.InputTag("goodZToMuMuAtLeast1HLTLoose")
)

goodZToMuMuPathLoose.__iadd__(goodZToMuMuVtxedAtLeast1HLTLoose)
goodZToMuMuPathLoose.setLabel("goodZToMuMuLoose")


goodZToMuMuVtxed2HLTLoose = copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTLoose)
goodZToMuMuVtxed2HLTLoose.src = cms.InputTag("goodZToMuMu2HLTLoose")
goodZToMuMu2HLTPathLoose.__iadd__(goodZToMuMuVtxed2HLTLoose)
goodZToMuMu2HLTPathLoose.setLabel("goodZToMuMu2HLTLoose")


goodZToMuMuVtxed1HLTLoose = copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTLoose)
goodZToMuMuVtxed1HLTLoose.src = cms.InputTag("goodZToMuMu1HLTLoose")
goodZToMuMu1HLTPathLoose.__iadd__(goodZToMuMuVtxed1HLTLoose)
goodZToMuMu1HLTPathLoose.setLabel("goodZToMuMu1HLTLoose")


goodZToMuMuVtxedBB2HLTLoose = copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTLoose)
goodZToMuMuVtxedBB2HLTLoose.src = cms.InputTag("goodZToMuMuBB2HLTLoose")
goodZToMuMuBB2HLTPathLoose.__iadd__(goodZToMuMuVtxedBB2HLTLoose)
goodZToMuMuBB2HLTPathLoose.setLabel("goodZToMuMuBB2HLTLoose")


goodZToMuMuVtxedAB1HLTLoose = copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTLoose)
goodZToMuMuVtxedAB1HLTLoose.src = cms.InputTag("goodZToMuMuAB1HLTLoose")
goodZToMuMuAB1HLTPathLoose.__iadd__(goodZToMuMuVtxedAB1HLTLoose)
goodZToMuMuAB1HLTPathLoose.setLabel("goodZToMuMuAB1HLTLoose")




## oneNonIsolatedZToMuMuVtxed= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLT)
## oneNonIsolatedZToMuMuVtxed.src= cms.InputTag("oneNonIsolatedZToMuMuAtLeast1HLT")
## oneNonIsolatedZToMuMuPath.__iadd__(oneNonIsolatedZToMuMuVtxed)
## oneNonIsolatedZToMuMuPath.setLabel("oneNonIsolatedZToMuMu")

## twoNonIsolatedZToMuMuVtxed = copy.deepcopy(goodZToMuMuVtxedAtLeast1HLT)
## twoNonIsolatedZToMuMuVtxed.src = cms.InputTag("twoNonIsolatedZToMuMuAtLeast1HLT")
## twoNonIsolatedZToMuMuPath.__iadd__(twoNonIsolatedZToMuMuVtxed)
## twoNonIsolatedZToMuMuPath.setLabel("twoNonIsolatedZToMuMu")

## goodZToMuMuSameCharge2HLTVtxedLoose= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTLoose)
## goodZToMuMuSameCharge2HLTVtxedLoose.src = cms.InputTag("goodZToMuMuSameCharge2HLTLoose")
## goodZToMuMuSameCharge2HLTPathLoose.__iadd__(goodZToMuMuSameCharge2HLTVtxedLoose)
## goodZToMuMuSameCharge2HLTPathLoose.setLabel("goodZToMuMuSameCharge2HLTLoose")


## goodZToMuMuSameCharge1HLTVtxedLoose= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTLoose)
## goodZToMuMuSameCharge1HLTVtxedLoose.src = cms.InputTag("goodZToMuMuSameCharge1HLTLoose")
## goodZToMuMuSameCharge1HLTPathLoose.__iadd__(goodZToMuMuSameCharge1HLTVtxedLoose)
## goodZToMuMuSameCharge1HLTPathLoose.setLabel("goodZToMuMuSameCharge1HLTLoose")


goodZToMuMuSameChargeVtxedLoose= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTLoose)
goodZToMuMuSameChargeVtxedLoose.src = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLTLoose")
goodZToMuMuSameChargePathLoose.__iadd__(goodZToMuMuSameChargeVtxedLoose)
goodZToMuMuSameChargePathLoose.setLabel("goodZToMuMuSameChargeLoose")



goodZToMuMuOneStandAloneVtxedLoose= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTLoose)
goodZToMuMuOneStandAloneVtxedLoose.src = cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLTLoose")
goodZToMuMuOneStandAloneMuonPathLoose.__iadd__(goodZToMuMuOneStandAloneVtxedLoose)
goodZToMuMuOneStandAloneMuonPathLoose.setLabel("goodZToMuMuOneStandAloneMuonLoose")

goodZToMuMuOneTrackVtxedLoose= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTLoose)
goodZToMuMuOneTrackVtxedLoose.src = cms.InputTag("goodZToMuMuOneTrackFirstHLTLoose")
goodZToMuMuOneTrackPathLoose.__iadd__(goodZToMuMuOneTrackVtxedLoose)
goodZToMuMuOneTrackPathLoose.setLabel("goodZToMuMuOneTrackLoose")

goodZToMuMuOneTrackerMuonVtxedLoose= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTLoose)
goodZToMuMuOneTrackerMuonVtxedLoose.src = cms.InputTag("goodZToMuMuOneTrackerMuonFirstHLTLoose")
goodZToMuMuOneTrackerMuonPathLoose.__iadd__(goodZToMuMuOneTrackerMuonVtxedLoose)
goodZToMuMuOneTrackerMuonPathLoose.setLabel("goodZToMuMuOneTrackerMuonLoose")




### ntuples....

goodZToMuMuVtxedNtupleLoose = cms.EDProducer(
    "CandViewNtpProducer",
    src = cms.InputTag("goodZToMuMuVtxedLoose"),
    variables = cms.VPSet(
      cms.PSet(
        tag = cms.untracked.string("mass"),
        quantity = cms.untracked.string("mass")
      ),
       cms.PSet(
        tag = cms.untracked.string("vertexNdof"),
        quantity = cms.untracked.string("vertexNdof")
      ),
       cms.PSet(
        tag = cms.untracked.string("vertexNormalizedChi2"),
        quantity = cms.untracked.string("vertexNormalizedChi2")
      ),
    )
)


goodZToMuMuVtxed2HLTNtupleLoose = copy.deepcopy(goodZToMuMuVtxedNtupleLoose)
goodZToMuMuVtxed2HLTNtupleLoose.src= cms.InputTag("goodZToMuMuVtxed2HLTLoose")
goodZToMuMu2HLTPathLoose.__iadd__(goodZToMuMuVtxed2HLTNtupleLoose)
goodZToMuMu2HLTPathLoose.setLabel("goodZToMuMu2HLTLoose")


goodZToMuMuVtxed1HLTNtupleLoose = copy.deepcopy(goodZToMuMuVtxedNtupleLoose)
goodZToMuMuVtxed1HLTNtupleLoose.src= cms.InputTag("goodZToMuMuVtxed1HLTLoose")
goodZToMuMu1HLTPathLoose.__iadd__(goodZToMuMuVtxed1HLTNtupleLoose)
goodZToMuMu1HLTPathLoose.setLabel("goodZToMuMu1HLTLoose")

goodZToMuMuVtxedBB2HLTNtupleLoose = copy.deepcopy(goodZToMuMuVtxedNtupleLoose)
goodZToMuMuVtxedBB2HLTNtupleLoose.src= cms.InputTag("goodZToMuMuVtxedBB2HLTLoose")
goodZToMuMuBB2HLTPathLoose.__iadd__(goodZToMuMuVtxedBB2HLTNtupleLoose)
goodZToMuMuBB2HLTPathLoose.setLabel("goodZToMuMuBB2HLTLoose")


goodZToMuMuVtxedAB1HLTNtupleLoose = copy.deepcopy(goodZToMuMuVtxedNtupleLoose)
goodZToMuMuVtxedAB1HLTNtupleLoose.src= cms.InputTag("goodZToMuMuVtxedAB1HLTLoose")
goodZToMuMuAB1HLTPathLoose.__iadd__(goodZToMuMuVtxedAB1HLTNtupleLoose)
goodZToMuMuAB1HLTPathLoose.setLabel("goodZToMuMuAB1HLTLoose")



## oneNonIsolatedZToMuMuVtxedNtuple = copy.deepcopy(goodZToMuMuVtxedNtuple)
## oneNonIsolatedZToMuMuVtxedNtuple.src = cms.InputTag("oneNonIsolatedZToMuMuVtxed")
## oneNonIsolatedZToMuMuPath.__iadd__(oneNonIsolatedZToMuMuVtxedNtuple)
## oneNonIsolatedZToMuMuPath.setLabel("oneNonIsolatedZToMuMu")

## twoNonIsolatedZToMuMuVtxedNtuple = copy.deepcopy(goodZToMuMuVtxedNtuple)
## twoNonIsolatedZToMuMuVtxedNtuple.src = cms.InputTag("twoNonIsolatedZToMuMuVtxed")
## twoNonIsolatedZToMuMuPath.__iadd__(twoNonIsolatedZToMuMuVtxedNtuple)
## twoNonIsolatedZToMuMuPath.setLabel("twoNonIsolatedZToMuMu")

## goodZToMuMuVtxedSameCharge2HLTNtupleLoose= copy.deepcopy(goodZToMuMuVtxedNtupleLoose)
## goodZToMuMuVtxedSameCharge2HLTNtupleLoose.src = cms.InputTag("goodZToMuMuVtxedSameCharge2HLTLoose")
## goodZToMuMuSameCharge2HLTPathLoose.__iadd__(goodZToMuMuVtxedSameCharge2HLTNtupleLoose)
## goodZToMuMuSameCharge2HLTPathLoose.setLabel("goodZToMuMuVtxedSameCharge2HLTLoose")


## goodZToMuMuVtxedSameCharge1HLTNtupleLoose= copy.deepcopy(goodZToMuMuVtxedNtupleLoose)
## goodZToMuMuVtxedSameCharge1HLTNtupleLoose.src =  cms.InputTag("goodZToMuMuVtxedSameCharge1HLTLoose")
## goodZToMuMuSameCharge1HLTPathLoose.__iadd__(goodZToMuMuVtxedSameCharge1HLTNtupleLoose)
## goodZToMuMuSameCharge1HLTPathLoose.setLabel("goodZToMuMuSameCharge1HLTLoose")




goodZToMuMuVtxedSameChargeNtupleLoose= copy.deepcopy(goodZToMuMuVtxedNtupleLoose)
goodZToMuMuVtxedSameChargeNtupleLoose.src =  cms.InputTag("goodZToMuMuSameChargeAtLeast1HLTLoose")
goodZToMuMuSameChargePathLoose.__iadd__(goodZToMuMuVtxedSameChargeNtupleLoose)
goodZToMuMuSameChargePathLoose.setLabel("goodZToMuMuSameChargeLoose")


goodZToMuMuVtxedOneStandAloneNtupleLoose= copy.deepcopy(goodZToMuMuVtxedNtupleLoose)
goodZToMuMuVtxedOneStandAloneNtupleLoose.src = cms.InputTag("goodZToMuMuOneStandAloneVtxedLoose")
goodZToMuMuOneStandAloneMuonPathLoose.__iadd__(goodZToMuMuVtxedOneStandAloneNtupleLoose)
goodZToMuMuOneStandAloneMuonPathLoose.setLabel("goodZToMuMuOneStandAloneMuonLoose")

goodZToMuMuVtxedOneTrackNtupleLoose= copy.deepcopy(goodZToMuMuVtxedNtupleLoose)
goodZToMuMuVtxedOneTrackNtupleLoose.src =cms.InputTag("goodZToMuMuOneTrackVtxedLoose")
goodZToMuMuOneTrackPathLoose.__iadd__(goodZToMuMuVtxedOneTrackNtupleLoose)
goodZToMuMuOneTrackPathLoose.setLabel("goodZToMuMuOneTrackLoose")


goodZToMuMuVtxedOneTrackerMuonNtupleLoose= copy.deepcopy(goodZToMuMuVtxedNtupleLoose)
goodZToMuMuVtxedOneTrackerMuonNtupleLoose.src =cms.InputTag("goodZToMuMuOneTrackerMuonVtxedLoose")
goodZToMuMuOneTrackerMuonPathLoose.__iadd__(goodZToMuMuVtxedOneTrackerMuonNtupleLoose)
goodZToMuMuOneTrackerMuonPathLoose.setLabel("goodZToMuMuOneTrackerMuonLoose")



vtxedNtuplesOut = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('VtxedNtupleLoose_test.root'),
    outputCommands = cms.untracked.vstring(
      "drop *",
#      "keep *_goodZToMuMuOneStandAloneMuonNtuple_*_*",
      "keep *_goodZToMuMuVtxedNtupleLoose_*_*",
      "keep *_goodZToMuMuVtxed1HLTNtupleLoose_*_*",
      "keep *_goodZToMuMuVtxed2HLTNtupleLoose_*_*",
      "keep *_goodZToMuMuVtxedAB1HLTNtupleLoose_*_*",
      "keep *_goodZToMuMuVtxedBB2HLTNtupleLoose_*_*",
#      "keep *_goodZToMuMuVtxedSameCharge2HLTNtupleLoose_*_*",
      "keep *_goodZToMuMuVtxedSameChargeNtupleLoose_*_*",
#      "keep *_nonIsolatedZToMuMuVtxedNtuple_*_*",
#      "keep *_oneNonIsolatedZToMuMuVtxedNtuple_*_*",
#      "keep *_twoNonIsolatedZToMuMuVtxedNtuple_*_*",
      "keep *_goodZToMuMuVtxedOneStandAloneNtupleLoose_*_*",
      "keep *_goodZToMuMuVtxedOneTrackNtupleLoose_*_*",
      "keep *_goodZToMuMuVtxedOneTrackerMuonNtupleLoose_*_*",
 #     "keep *_goodZToMuMu2HLTVtxedNtuple_*_*",
      
    ),
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring(
        "goodZToMuMuPathLoose",
        "goodZToMuMu1HLTPathLoose",
        "goodZToMuMu2HLTPathLoose",
        "goodZToMuMuAB1HLTPathLoose",
        "goodZToMuMuBB2HLTPathLoose",
#        "goodZToMuMuSameCharge2HLTPathLoose",
        "goodZToMuMuSameChargePathLoose",
 #       "nonIsolatedZToMuMuPath",
 #       "oneNonIsolatedZToMuMuPath",
 #       "twoNonIsolatedZToMuMuPath",
        "goodZToMuMuOneTrackPathLoose",
        "goodZToMuMuOneTrackerMuonPathLoose",
        "goodZToMuMuOneStandAloneMuonPathLoose",
      )
    )
)


vtxedNtuplesOut.setLabel("vtxedNtuplesOut")
VtxedNtuplesOut.__iadd__(vtxedNtuplesOut)
VtxedNtuplesOut.setLabel("VtxedNtuplesOut")


## ## vertex refit for tight cut

## goodZToMuMuVtxedAtLeast1HLTTight = cms.EDProducer(
##     "KalmanVertexFitCompositeCandProducer",
##     src = cms.InputTag("goodZToMuMuAtLeast1HLTTight")
## )


## goodZToMuMuVtxed2HLTTight = copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTTight)
## goodZToMuMuVtxed2HLTTight.src = cms.InputTag("goodZToMuMu2HLTTight")
## goodZToMuMu2HLTPathTight.__iadd__(goodZToMuMuVtxed2HLTTight)
## goodZToMuMu2HLTPathTight.setLabel("goodZToMuMu2HLTTight")


## goodZToMuMuVtxed1HLTTight = copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTTight)
## goodZToMuMuVtxed1HLTTight.src = cms.InputTag("goodZToMuMu1HLTTight")
## goodZToMuMu1HLTPathTight.__iadd__(goodZToMuMuVtxed1HLTTight)
## goodZToMuMu1HLTPathTight.setLabel("goodZToMuMu1HLTTight")

## oneNonIsolatedZToMuMuVtxedTight= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTTight)
## oneNonIsolatedZToMuMuVtxedTight.src= cms.InputTag("oneNonIsolatedZToMuMuAtLeast1HLTTight")
## oneNonIsolatedZToMuMuPathTight.__iadd__(oneNonIsolatedZToMuMuVtxedTight)
## oneNonIsolatedZToMuMuPathTight.setLabel("oneNonIsolatedZToMuMuTight")

## twoNonIsolatedZToMuMuVtxedTight = copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTTight)
## twoNonIsolatedZToMuMuVtxedTight.src = cms.InputTag("twoNonIsolatedZToMuMuAtLeast1HLTTight")
## twoNonIsolatedZToMuMuPathTight.__iadd__(twoNonIsolatedZToMuMuVtxedTight)
## twoNonIsolatedZToMuMuPathTight.setLabel("twoNonIsolatedZToMuMuTight")

## goodZToMuMuSameCharge2HLTVtxedTight= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTTight)
## goodZToMuMuSameCharge2HLTVtxedTight.src = cms.InputTag("goodZToMuMuSameCharge2HLTTight")
## goodZToMuMuSameCharge2HLTPathTight.__iadd__(goodZToMuMuSameCharge2HLTVtxedTight)
## goodZToMuMuSameCharge2HLTPathTight.setLabel("goodZToMuMuSameCharge2HLTTight")


## goodZToMuMuSameCharge1HLTVtxedTight= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTTight)
## goodZToMuMuSameCharge1HLTVtxedTight.src = cms.InputTag("goodZToMuMuSameCharge1HLTTight")
## goodZToMuMuSameCharge1HLTPathTight.__iadd__(goodZToMuMuSameCharge1HLTVtxedTight)
## goodZToMuMuSameCharge1HLTPathTight.setLabel("goodZToMuMuSameCharge1HLTTight")



## goodZToMuMuOneStandAloneVtxedTight= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTTight)
## goodZToMuMuOneStandAloneVtxedTight.src = cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLTTight")
## goodZToMuMuOneStandAloneMuonPathTight.__iadd__(goodZToMuMuOneStandAloneVtxedTight)
## goodZToMuMuOneStandAloneMuonPathTight.setLabel("goodZToMuMuOneStandAloneMuonTight")

## goodZToMuMuOneTrackVtxedTight= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLTTight)
## goodZToMuMuOneTrackVtxedTight.src = cms.InputTag("goodZToMuMuOneTrackFirstHLTTight")
## goodZToMuMuOneTrackPathTight.__iadd__(goodZToMuMuOneTrackVtxedTight)
## goodZToMuMuOneTrackPathTight.setLabel("goodZToMuMuOneTrackTight")



## ### ntuples....

## goodZToMuMuVtxedNtupleTight = cms.EDProducer(
##     "CandViewNtpProducer",
##     src = cms.InputTag("goodZToMuMuVtxedTight"),
##     variables = cms.VPSet(
##       cms.PSet(
##         tag = cms.untracked.string("mass"),
##         quantity = cms.untracked.string("mass")
##       ),
##        cms.PSet(
##         tag = cms.untracked.string("vertexNdof"),
##         quantity = cms.untracked.string("vertexNdof")
##       ),
##        cms.PSet(
##         tag = cms.untracked.string("vertexNormalizedChi2"),
##         quantity = cms.untracked.string("vertexNormalizedChi2")
##       ),
##     )
## )


## goodZToMuMuVtxed2HLTNtupleTight = copy.deepcopy(goodZToMuMuVtxedNtupleTight)
## goodZToMuMuVtxed2HLTNtupleTight.src= cms.InputTag("goodZToMuMuVtxed2HLTTight")
## goodZToMuMu2HLTPathTight.__iadd__(goodZToMuMuVtxed2HLTTightNtupleTight)
## goodZToMuMu2HLTPathTight.setLabel("goodZToMuMu2HLTTight")


## goodZToMuMuVtxed1HLTNtupleTight = copy.deepcopy(goodZToMuMuVtxedNtupleTight)
## goodZToMuMuVtxed1HLTNtupleTight.src= cms.InputTag("goodZToMuMuVtxed1HLTTight")
## goodZToMuMu1HLTPathTight.__iadd__(goodZToMuMuVtxed1HLTNtupleTight)
## goodZToMuMu1HLTPathTight.setLabel("goodZToMuMu1HLTTight")

## oneNonIsolatedZToMuMuVtxedNtupleTight = copy.deepcopy(goodZToMuMuVtxedNtupleTight)
## oneNonIsolatedZToMuMuVtxedNtupleTight.src = cms.InputTag("oneNonIsolatedZToMuMuVtxedTight")
## oneNonIsolatedZToMuMuPathTight.__iadd__(oneNonIsolatedZToMuMuVtxedNtupleTight)
## oneNonIsolatedZToMuMuPathTight.setLabel("oneNonIsolatedZToMuMuTight")

## twoNonIsolatedZToMuMuVtxedNtupleTight = copy.deepcopy(goodZToMuMuVtxedNtupleTight)
## twoNonIsolatedZToMuMuVtxedNtupleTight.src = cms.InputTag("twoNonIsolatedZToMuMuVtxed")
## twoNonIsolatedZToMuMuPathTight.__iadd__(twoNonIsolatedZToMuMuVtxedNtupleTight)
## twoNonIsolatedZToMuMuPathTight.setLabel("twoNonIsolatedZToMuMuTight")

## goodZToMuMuVtxedSameCharge2HLTNtupleTight= copy.deepcopy(goodZToMuMuVtxedNtupleTight)
## goodZToMuMuVtxedSameCharge2HLTNtupleTight.src = cms.InputTag("goodZToMuMuVtxedSameCharge2HLTTight")
## goodZToMuMuSameCharge2HLTPathTight.__iadd__(goodZToMuMuVtxedSameCharge2HLTNtupleTight)
## goodZToMuMuSameCharge2HLTPathTight.setLabel("goodZToMuMuVtxedSameCharge2HLTTight")


## goodZToMuMuVtxedSameCharge1HLTNtupleTight= copy.deepcopy(goodZToMuMuVtxedNtupleTight)
## goodZToMuMuVtxedSameCharge1HLTNtupleTight.src =  cms.InputTag("goodZToMuMuVtxedSameCharge1HLTTight")
## goodZToMuMuSameCharge1HLTPathTight.__iadd__(goodZToMuMuVtxedSameCharge1HLTNtupleTight)
## goodZToMuMuSameCharge1HLTPathTight.setLabel("goodZToMuMuSameCharge1HLTTight")


## goodZToMuMuVtxedOneStandAloneNtupleTight= copy.deepcopy(goodZToMuMuVtxedNtupleTight)
## goodZToMuMuVtxedOneStandAloneNtupleTight.src = cms.InputTag("goodZToMuMuOneStandAloneVtxedTight")
## goodZToMuMuOneStandAloneMuonPathTight.__iadd__(goodZToMuMuVtxedOneStandAloneNtupleTight)
## goodZToMuMuOneStandAloneMuonPathTight.setLabel("goodZToMuMuOneStandAloneMuonTight")

## goodZToMuMuVtxedOneTrackNtupleTight= copy.deepcopy(goodZToMuMuVtxedNtupleTight)
## goodZToMuMuVtxedOneTrackNtupleTight.src =cms.InputTag("goodZToMuMuOneTrackVtxed")
## goodZToMuMuOneTrackPathTight.__iadd__(goodZToMuMuVtxedOneTrackNtupleTight)
## goodZToMuMuOneTrackPathTight.setLabel("goodZToMuMuOneTrackTight")



## vtxedNtuplesOutTight = cms.OutputModule(
##     "PoolOutputModule",
##     fileName = cms.untracked.string('VtxedNtupleTight_test.root'),
##     outputCommands = cms.untracked.vstring(
##       "drop *",
## #      "keep *_goodZToMuMuOneStandAloneMuonNtuple_*_*",
##       "keep *_goodZToMuMuVtxedNtupleTight_*_*",
##       "keep *_goodZToMuMuVtxed1HLTNtupleTight_*_*",
##       "keep *_goodZToMuMuVtxed2HLTNtupleTight_*_*",
##       "keep *_goodZToMuMuVtxedSameCharge2HLTNtupleTight_*_*",
##       "keep *_goodZToMuMuVtxedSameCharge1HLTNtupleTight_*_*",
##       "keep *_nonIsolatedZToMuMuVtxedNtupleTight_*_*",
##       "keep *_oneNonIsolatedZToMuMuVtxedNtupleTight_*_*",
##       "keep *_twoNonIsolatedZToMuMuVtxedNtupleTight_*_*",
##       "keep *_goodZToMuMuVtxedOneStandAloneNtupleTight_*_*",
##       "keep *_goodZToMuMuVtxedOneTrackNtupleTight_*_*",
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


## vtxedNtuplesOutTight.setLabel("vtxedNtuplesOutTight")
## VtxedNtuplesOutTight.__iadd__(vtxedNtuplesOutTight)
## VtxedNtuplesOutTight.setLabel("VtxedNtuplesOutTight")
