import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *
import copy

goodZToMuMuVtxedAtLeast1HLT = cms.EDProducer(
    "KalmanVertexFitCompositeCandProducer",
    src = cms.InputTag("goodZToMuMuAtLeast1HLT")
)


goodZToMuMuVtxed2HLT = copy.deepcopy(goodZToMuMuVtxedAtLeast1HLT)
goodZToMuMuVtxed2HLT.src = cms.InputTag("goodZToMuMu2HLT")
goodZToMuMu2HLTSequence.__iadd__(goodZToMuMuVtxed2HLT)
goodZToMuMu2HLTSequence.setLabel("goodZToMuMu2HLT")


goodZToMuMuVtxed1HLT = copy.deepcopy(goodZToMuMuVtxedAtLeast1HLT)
goodZToMuMuVtxed1HLT.src = cms.InputTag("goodZToMuMu1HLT")
goodZToMuMu1HLTSequence.__iadd__(goodZToMuMuVtxed1HLT)
goodZToMuMu1HLTSequence.setLabel("goodZToMuMu1HLT")

oneNonIsolatedZToMuMuVtxed= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLT)
oneNonIsolatedZToMuMuVtxed.src= cms.InputTag("oneNonIsolatedZToMuMuAtLeast1HLT")
oneNonIsolatedZToMuMuSequence.__iadd__(oneNonIsolatedZToMuMuVtxed)
oneNonIsolatedZToMuMuSequence.setLabel("oneNonIsolatedZToMuMu")

twoNonIsolatedZToMuMuVtxed = copy.deepcopy(goodZToMuMuVtxedAtLeast1HLT)
twoNonIsolatedZToMuMuVtxed.src = cms.InputTag("twoNonIsolatedZToMuMuAtLeast1HLT")
twoNonIsolatedZToMuMuSequence.__iadd__(twoNonIsolatedZToMuMuVtxed)
twoNonIsolatedZToMuMuSequence.setLabel("twoNonIsolatedZToMuMu")

goodZToMuMuSameCharge2HLTVtxed= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLT)
goodZToMuMuSameCharge2HLTVtxed.src = cms.InputTag("goodZToMuMuSameCharge2HLT")
goodZToMuMuSameCharge2HLTSequence.__iadd__(goodZToMuMuSameCharge2HLTVtxed)
goodZToMuMuSameCharge2HLTSequence.setLabel("goodZToMuMuSameCharge2HLT")


goodZToMuMuSameCharge1HLTVtxed= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLT)
goodZToMuMuSameCharge1HLTVtxed.src = cms.InputTag("goodZToMuMuSameCharge1HLT")
goodZToMuMuSameCharge1HLTSequence.__iadd__(goodZToMuMuSameCharge1HLTVtxed)
goodZToMuMuSameCharge1HLTSequence.setLabel("goodZToMuMuSameCharge1HLT")



goodZToMuMuOneStandAloneVtxed= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLT)
goodZToMuMuOneStandAloneVtxed.src = cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLT")
goodZToMuMuOneStandAloneMuonSequence.__iadd__(goodZToMuMuOneStandAloneVtxed)
goodZToMuMuOneStandAloneMuonSequence.setLabel("goodZToMuMuOneStandAloneMuon")

goodZToMuMuOneTrackVtxed= copy.deepcopy(goodZToMuMuVtxedAtLeast1HLT)
goodZToMuMuOneTrackVtxed.src = cms.InputTag("goodZToMuMuOneTrackFirstHLT")
goodZToMuMuOneTrackSequence.__iadd__(goodZToMuMuOneTrackVtxed)
goodZToMuMuOneTrackSequence.setLabel("goodZToMuMuOneTrack")



### ntuples....

goodZToMuMuVtxedNtuple = cms.EDProducer(
    "CandViewNtpProducer",
    src = cms.InputTag("goodZToMuMuVtxed"),
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


goodZToMuMuVtxed2HLTNtuple = copy.deepcopy(goodZToMuMuVtxedNtuple)
goodZToMuMuVtxed2HLTNtuple.src= cms.InputTag("goodZToMuMuVtxed2HLT")
goodZToMuMu2HLTSequence.__iadd__(goodZToMuMuVtxed2HLTNtuple)
goodZToMuMu2HLTSequence.setLabel("goodZToMuMu2HLT")


goodZToMuMuVtxed1HLTNtuple = copy.deepcopy(goodZToMuMuVtxedNtuple)
goodZToMuMuVtxed1HLTNtuple.src= cms.InputTag("goodZToMuMuVtxed1HLT")
goodZToMuMu1HLTSequence.__iadd__(goodZToMuMuVtxed1HLTNtuple)
goodZToMuMu1HLTSequence.setLabel("goodZToMuMu1HLT")

oneNonIsolatedZToMuMuVtxedNtuple = copy.deepcopy(goodZToMuMuVtxedNtuple)
oneNonIsolatedZToMuMuVtxedNtuple.src = cms.InputTag("oneNonIsolatedZToMuMuVtxed")
oneNonIsolatedZToMuMuSequence.__iadd__(oneNonIsolatedZToMuMuVtxedNtuple)
oneNonIsolatedZToMuMuSequence.setLabel("oneNonIsolatedZToMuMu")

twoNonIsolatedZToMuMuVtxedNtuple = copy.deepcopy(goodZToMuMuVtxedNtuple)
twoNonIsolatedZToMuMuVtxedNtuple.src = cms.InputTag("twoNonIsolatedZToMuMuVtxed")
twoNonIsolatedZToMuMuSequence.__iadd__(twoNonIsolatedZToMuMuVtxedNtuple)
twoNonIsolatedZToMuMuSequence.setLabel("twoNonIsolatedZToMuMu")

goodZToMuMuVtxedSameCharge2HLTNtuple= copy.deepcopy(goodZToMuMuVtxedNtuple)
goodZToMuMuVtxedSameCharge2HLTNtuple.src = cms.InputTag("goodZToMuMuVtxedSameCharge2HLT")
goodZToMuMuSameCharge2HLTSequence.__iadd__(goodZToMuMuVtxedSameCharge2HLTNtuple)
goodZToMuMuSameCharge2HLTSequence.setLabel("goodZToMuMuVtxedSameCharge2HLT")


goodZToMuMuVtxedSameCharge1HLTNtuple= copy.deepcopy(goodZToMuMuVtxedNtuple)
goodZToMuMuVtxedSameCharge1HLTNtuple.src =  cms.InputTag("goodZToMuMuVtxedSameCharge1HLT")
goodZToMuMuSameCharge1HLTSequence.__iadd__(goodZToMuMuVtxedSameCharge1HLTNtuple)
goodZToMuMuSameCharge1HLTSequence.setLabel("goodZToMuMuSameCharge1HLT")


goodZToMuMuVtxedOneStandAloneNtuple= copy.deepcopy(goodZToMuMuVtxedNtuple)
goodZToMuMuVtxedOneStandAloneNtuple.src = cms.InputTag("goodZToMuMuOneStandAloneVtxed")
goodZToMuMuOneStandAloneMuonSequence.__iadd__(goodZToMuMuVtxedOneStandAloneNtuple)
goodZToMuMuOneStandAloneMuonSequence.setLabel("goodZToMuMuOneStandAloneMuon")

goodZToMuMuVtxedOneTrackNtuple= copy.deepcopy(goodZToMuMuVtxedNtuple)
goodZToMuMuVtxedOneTrackNtuple.src =cms.InputTag("goodZToMuMuOneTrackVtxed")
goodZToMuMuOneTrackSequence.__iadd__(goodZToMuMuVtxedOneTrackNtuple)
goodZToMuMuOneTrackSequence.setLabel("goodZToMuMuOneTrack")



VtxedNtuplesOut = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('VtxedNtuple_test.root'),
    outputCommands = cms.untracked.vstring(
      "drop *",
#      "keep *_goodZToMuMuOneStandAloneMuonNtuple_*_*",
      "keep *_goodZToMuMuVtxedNtuple_*_*",
      "keep *_goodZToMuMuVtxed1HLTNtuple_*_*",
      "keep *_goodZToMuMuVtxed2HLTNtuple_*_*",
      "keep *_goodZToMuMuVtxedSameCharge2HLTNtuple_*_*",
      "keep *_goodZToMuMuVtxedSameCharge1HLTNtuple_*_*",
      "keep *_nonIsolatedZToMuMuVtxedNtuple_*_*",
      "keep *_oneNonIsolatedZToMuMuVtxedNtuple_*_*",
      "keep *_twoNonIsolatedZToMuMuVtxedNtuple_*_*",
      "keep *_goodZToMuMuVtxedOneStandAloneNtuple_*_*",
      "keep *_goodZToMuMuVtxedOneTrackNtuple_*_*",
 #     "keep *_goodZToMuMu2HLTVtxedNtuple_*_*",
      
    ),
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring(
        "goodZToMuMuPath",
        "goodZToMuMu1HLTPath",
        "goodZToMuMu2HLTPath",
        "goodZToMuMuSameCharge2HLTPath",
        "goodZToMuMuSameCharge1HLTPath",
        "nonIsolatedZToMuMuPath",
        "oneNonIsolatedZToMuMuPath",
        "twoNonIsolatedZToMuMuPath",
        "goodZToMuMuOneTrackPath",
        "goodZToMuMuOneStandAloneMuonPath",
      )
    )
)
