import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *

import copy

zGolden=(
    cms.PSet(
    tag = cms.untracked.string("Dau2NofHit"),
    quantity = cms.untracked.string("daughter(1).masterClone.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2NofHitTk"),
    quantity = cms.untracked.string("daughter(1).masterClone.innerTrack.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2NofHitSta"),
    quantity = cms.untracked.string("daughter(1).masterClone.outerTrack.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2NofMuChambers"),
    quantity = cms.untracked.string("daughter(1).masterClone.numberOfChambers")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2NofMuMatches"),
    quantity = cms.untracked.string("daughter(1).masterClone.numberOfMatches")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Chi2"),
    quantity = cms.untracked.string("daughter(1).masterClone.normChi2")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2dB"),
    quantity = cms.untracked.string("daughter(1).masterClone.dB")
    )
    )

zMuTrk=(
    
    )

goodZToMuMuEdmNtuple = cms.EDProducer(
    "CandViewNtpProducer", 
    src=cms.InputTag("goodZToMuMuAtLeast1HLTLoose"),
    lazyParser=cms.untracked.bool(True),
    prefix=cms.untracked.string("z"),
    eventInfo=cms.untracked.bool(True),
    variables = cms.VPSet(
    cms.PSet(
    tag = cms.untracked.string("Mass"),
    quantity = cms.untracked.string("mass")
    ),
    cms.PSet(
    tag = cms.untracked.string("Pt"),
    quantity = cms.untracked.string("pt")
    ),
    cms.PSet(
    tag = cms.untracked.string("Eta"),
    quantity = cms.untracked.string("eta")
    ),
    cms.PSet(
    tag = cms.untracked.string("Phi"),
    quantity = cms.untracked.string("phi")
    ),
    cms.PSet(
    tag = cms.untracked.string("Y"),
    quantity = cms.untracked.string("rapidity")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1Pt"),
    quantity = cms.untracked.string("daughter(0).masterClone.pt")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Pt"),
    quantity = cms.untracked.string("daughter(1).masterClone.pt")
    ),
    #cms.PSet(
    #tag = cms.untracked.string("Dau1HLTBit"),
    #quantity = cms.untracked.string("daughter(0).masterClone.pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau2HLTBit"),
    #quantity = cms.untracked.string("pt")
    #),
    cms.PSet(
    tag = cms.untracked.string("Dau1Q"),
    quantity = cms.untracked.string("daughter(0).masterClone.charge")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Q"),
    quantity = cms.untracked.string("daughter(1).masterClone.charge")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1Eta"),
    quantity = cms.untracked.string("daughter(0).masterClone.eta")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Eta"),
    quantity = cms.untracked.string("daughter(1).masterClone.eta")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1Phi"),
    quantity = cms.untracked.string("daughter(0).masterClone.phi")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Phi"),
    quantity = cms.untracked.string("daughter(1).masterClone.phi")
    ),
    #cms.PSet(
    #tag = cms.untracked.string("Dau1Iso"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau2Iso"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau1TrkIso"),
    #quantity = cms.untracked.string("daughter(0).masterClone.pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau2TrkIso"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau1EcalIso"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau2EcalIso"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau1HcalIso"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau2HcalIso"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),

    cms.PSet(
    tag = cms.untracked.string("Dau1NofHit"),
    quantity = cms.untracked.string("daughter(0).masterClone.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1NofHitTk"),
    quantity = cms.untracked.string("daughter(0).masterClone.innerTrack.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1NofHitSta"),
    quantity = cms.untracked.string("daughter(0).masterClone.outerTrack.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1NofMuChambers"),
    quantity = cms.untracked.string("daughter(0).masterClone.numberOfChambers")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1NofMuMatches"),
    quantity = cms.untracked.string("daughter(0).masterClone.numberOfMatches")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1Chi2"),
    quantity = cms.untracked.string("daughter(0).masterClone.normChi2")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau1dB"),
    quantity = cms.untracked.string("daughter(0).masterClone.dB")
    ),
    #cms.PSet(
    #tag = cms.untracked.string("Dau1dxyFromBS"),
    #quantity = cms.untracked.string("daughter(0).masterClone.innerTrack.dxy()")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau2dxyFromBS"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau1dzFromBS"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau2dzFromBS"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau1dxyFromPV"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau2dxyFromPV"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau1dzFromPV"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("Dau2dzFromPV"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("TrueMass"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("TruePt"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("TrueEta"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("TruePhi"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("TrueY"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
    #cms.PSet(
    #tag = cms.untracked.string("TruePt"),
    #quantity = cms.untracked.string("daughter(1).pt")
    #),
   
    )
    
    )


#### ntuple for loose cuts


goodZToMuMuEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuEdmNtupleLoose.variables += zGolden
goodZToMuMuEdmNtupleLoose.prefix = cms.untracked.string("zGolden")
goodZToMuMuPathLoose.__iadd__(goodZToMuMuEdmNtupleLoose)
goodZToMuMuPathLoose.setLabel("goodZToMuMuEdmLoose")


goodZToMuMuSameChargeEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuSameChargeEdmNtupleLoose.src = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLTLoose")
goodZToMuMuSameChargeEdmNtupleLoose.prefix = cms.untracked.string("zSameCharge")
goodZToMuMuSameChargeEdmNtupleLoose.variables += zGolden
#goodZToMuMuSameChargeEdmNtupleLoose.eventInfo = cms.untracked.bool(False)
goodZToMuMuSameChargePathLoose.__iadd__(goodZToMuMuSameChargeEdmNtupleLoose)
goodZToMuMuSameChargePathLoose.setLabel("goodZToMuMuSameChargeLoose")



goodZToMuMuOneStandAloneEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuOneStandAloneEdmNtupleLoose.src=cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLTLoose")
goodZToMuMuOneStandAloneEdmNtupleLoose.prefix=cms.untracked.string("zMuSta")
goodZToMuMuOneStandAloneEdmNtupleLoose.variables += zGolden
#goodZToMuMuOneStandAloneEdmNtupleLoose.eventInfo=cms.untracked.bool(False)
goodZToMuMuOneStandAloneMuonPathLoose.__iadd__(goodZToMuMuOneStandAloneEdmNtupleLoose)
goodZToMuMuOneStandAloneMuonPathLoose.setLabel("goodZToMuMuOneStandAloneMuonLoose")

goodZToMuMuOneTrackEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuOneTrackEdmNtupleLoose.src=cms.InputTag("goodZToMuMuOneTrackFirstHLTLoose")
goodZToMuMuOneTrackEdmNtupleLoose.prefix=cms.untracked.string("zMuTrk")
#goodZToMuMuOneTrackEdmNtupleLoose.eventInfo=cms.untracked.bool(False)
goodZToMuMuOneTrackPathLoose.__iadd__(goodZToMuMuOneTrackEdmNtupleLoose)
goodZToMuMuOneTrackPathLoose.setLabel("goodZToMuMuOneTrackLoose")

ntuplesOut = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('NtupleLoose_test.root'),
    outputCommands = cms.untracked.vstring(
      "drop *",
##      "keep *_goodZToMuMuOneStandAloneMuonNtuple_*_*",
      "keep *_goodZToMuMuEdmNtupleLoose_*_*",
##    "keep *_goodZToMuMu1HLTEdmNtupleLoose_*_*",
##    "keep *_goodZToMuMu2HLTEdmNtupleLoose_*_*",
      "keep *_goodZToMuMuSameChargeEdmNtupleLoose_*_*",
##      "keep *_goodZToMuMuSameCharge1HLTEdmNtupleLoose_*_*",
##   "keep *_nonIsolatedZToMuMuEdmNtuple_*_*",
##    "keep *_oneNonIsolatedZToMuMuEdmNtuple_*_*",
##    "keep *_twoNonIsolatedZToMuMuEdmNtuple_*_*",
      "keep *_goodZToMuMuOneStandAloneEdmNtupleLoose_*_*",
      "keep *_goodZToMuMuOneTrackEdmNtupleLoose_*_*"#,
##     "keep *_goodZToMuMu2HLTVtxedNtuple_*_*",
      
    )#,
#    SelectEvents = cms.untracked.PSet(
#      SelectEvents = cms.vstring(
#        "goodZToMuMuPathLoose",
##     "goodZToMuMu1HLTPathLoose",
##     "goodZToMuMu2HLTPathLoose",
#        "goodZToMuMuSameChargePathLoose",
##        "goodZToMuMuSameCharge1HLTPathLoose",
##     "nonIsolatedZToMuMuPath",
##     "oneNonIsolatedZToMuMuPath",
##     "twoNonIsolatedZToMuMuPath",
#        "goodZToMuMuOneTrackPathLoose",
#        "goodZToMuMuOneStandAloneMuonPathLoose",
#      )
#    )
)

ntuplesOut.setLabel("ntuplesOut")
NtuplesOut.__iadd__(ntuplesOut)
NtuplesOut.setLabel("NtuplesOut")


