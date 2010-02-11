import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *
import copy


goodZToMuMuEdmNtuple = cms.EDProducer(
    "ZToLLEdmNtupleDumper",
    zBlocks = cms.VPSet(
    cms.PSet(
        zName = cms.string("zGolden"),
        z = cms.InputTag("goodZToMuMuAtLeast1HLT"),
        zGenParticlesMatch = cms.InputTag(""),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        primaryVertices = cms.InputTag("offlinePrimaryVerticesWithBS"), 
        ptThreshold = cms.double("1.5"),
        etEcalThreshold = cms.double("0.2"),
        etHcalThreshold = cms.double("0.5"),
        deltaRVetoTrk = cms.double("0.015"),
        deltaRTrk = cms.double("0.3"),
        deltaREcal = cms.double("0.25"),
        deltaRHcal = cms.double("0.25"),
        alpha = cms.double("0."),
        beta = cms.double("-0.75"),
        relativeIsolation = cms.bool(False)
      ),
     )
)



goodZToMuMu2HLTEdmNtuple = copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMu2HLTEdmNtuple.zBlocks[0].z = cms.InputTag("goodZToMuMu2HLT")
goodZToMuMu2HLTEdmNtuple.zBlocks[0].zName = cms.string("zGolden2HLT")
goodZToMuMu2HLTSequence.__iadd__(goodZToMuMu2HLTEdmNtuple)
goodZToMuMu2HLTSequence.setLabel("goodZToMuMu2HLT")


goodZToMuMu1HLTEdmNtuple = copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMu1HLTEdmNtuple.zBlocks[0].z = cms.InputTag("goodZToMuMu1HLT")
goodZToMuMu1HLTEdmNtuple.zBlocks[0].zName = cms.string("zGolden1HLT")
goodZToMuMu1HLTSequence.__iadd__(goodZToMuMu1HLTEdmNtuple)
goodZToMuMu1HLTSequence.setLabel("goodZToMuMu1HLT")

oneNonIsolatedZToMuMuEdmNtuple = copy.deepcopy(goodZToMuMuEdmNtuple)
oneNonIsolatedZToMuMuEdmNtuple.zBlocks[0].z = cms.InputTag("oneNonIsolatedZToMuMuAtLeast1HLT")
oneNonIsolatedZToMuMuEdmNtuple.zBlocks[0].zName = cms.string("z1NotIso")
oneNonIsolatedZToMuMuSequence.__iadd__(oneNonIsolatedZToMuMuEdmNtuple)
oneNonIsolatedZToMuMuSequence.setLabel("oneNonIsolatedZToMuMu")

twoNonIsolatedZToMuMuEdmNtuple = copy.deepcopy(goodZToMuMuEdmNtuple)
twoNonIsolatedZToMuMuEdmNtuple.zBlocks[0].z = cms.InputTag("twoNonIsolatedZToMuMuAtLeast1HLT")
twoNonIsolatedZToMuMuEdmNtuple.zBlocks[0].zName = cms.string("z2NotIso")
twoNonIsolatedZToMuMuSequence.__iadd__(twoNonIsolatedZToMuMuEdmNtuple)
twoNonIsolatedZToMuMuSequence.setLabel("twoNonIsolatedZToMuMu")

goodZToMuMuSameCharge2HLTEdmNtuple= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuSameCharge2HLTEdmNtuple.zBlocks[0].z = cms.InputTag("goodZToMuMuSameCharge2HLT")
goodZToMuMuSameCharge2HLTEdmNtuple.zBlocks[0].zName = cms.string("zSameCharge2HLT")
goodZToMuMuSameCharge2HLTSequence.__iadd__(goodZToMuMuSameCharge2HLTEdmNtuple)
goodZToMuMuSameCharge2HLTSequence.setLabel("goodZToMuMuSameCharge2HLT")


goodZToMuMuSameCharge1HLTEdmNtuple= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuSameCharge1HLTEdmNtuple.zBlocks[0].z = cms.InputTag("goodZToMuMuSameCharge1HLT")
goodZToMuMuSameCharge1HLTEdmNtuple.zBlocks[0].zName = cms.string("zSameCharge1HLT")
goodZToMuMuSameCharge1HLTSequence.__iadd__(goodZToMuMuSameCharge1HLTEdmNtuple)
goodZToMuMuSameCharge1HLTSequence.setLabel("goodZToMuMuSameCharge1HLT")


goodZToMuMuOneStandAloneEdmNtuple= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuOneStandAloneEdmNtuple.zBlocks[0].z=cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLT")
goodZToMuMuOneStandAloneEdmNtuple.zBlocks[0].zName=cms.string("zMuSta")
goodZToMuMuOneStandAloneMuonSequence.__iadd__(goodZToMuMuOneStandAloneEdmNtuple)
goodZToMuMuOneStandAloneMuonSequence.setLabel("goodZToMuMuOneStandAloneMuon")

goodZToMuMuOneTrackEdmNtuple= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuOneTrackEdmNtuple.zBlocks[0].z=cms.InputTag("goodZToMuMuOneTrackFirstHLT")
goodZToMuMuOneTrackEdmNtuple.zBlocks[0].zName=cms.string("zMuTrk")
goodZToMuMuOneTrackSequence.__iadd__(goodZToMuMuOneTrackEdmNtuple)
goodZToMuMuOneTrackSequence.setLabel("goodZToMuMuOneTrack")

NtuplesOut = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('Ntuple_test.root'),
    outputCommands = cms.untracked.vstring(
      "drop *",
#      "keep *_goodZToMuMuOneStandAloneMuonNtuple_*_*",
      "keep *_goodZToMuMuEdmNtuple_*_*",
      "keep *_goodZToMuMu1HLTEdmNtuple_*_*",
      "keep *_goodZToMuMu2HLTEdmNtuple_*_*",
      "keep *_goodZToMuMuSameCharge2HLTEdmNtuple_*_*",
      "keep *_goodZToMuMuSameCharge1HLTEdmNtuple_*_*",
      "keep *_nonIsolatedZToMuMuEdmNtuple_*_*",
      "keep *_oneNonIsolatedZToMuMuEdmNtuple_*_*",
      "keep *_twoNonIsolatedZToMuMuEdmNtuple_*_*",
      "keep *_goodZToMuMuOneStandAloneEdmNtuple_*_*",
      "keep *_goodZToMuMuOneTrackEdmNtuple_*_*",
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
  

