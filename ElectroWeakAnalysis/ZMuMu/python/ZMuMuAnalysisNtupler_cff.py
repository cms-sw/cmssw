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
    )#,
    #cms.PSet(
    #tag = cms.untracked.string("Dau1UserIso"),
    #quantity = cms.untracked.string("daughter(0).masterClone.User1Iso")
    #)#,
    #cms.PSet(
    #tag = cms.untracked.string("Dau1UserIsolation"),
    #quantity = cms.untracked.string("daughter(0).masterClone.isolations_[7]")
    #)
    )

zMuTrk=(
    cms.PSet(
    tag = cms.untracked.string("Dau2NofHit"),
    quantity = cms.untracked.string("daughter(1).masterClone.track.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2NofHitTk"),
    quantity = cms.untracked.string("daughter(1).masterClone.track.numberOfValidHits")
    ),
    cms.PSet(
    tag = cms.untracked.string("Dau2Chi2"),
    quantity = cms.untracked.string("daughter(1).masterClone.track.normalizedChi2")
    )
    
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
    )
    )
    
    )



goodZToMuMuEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
goodZToMuMuEdmNtupleLoose.variables += zGolden
goodZToMuMuEdmNtupleLoose.prefix = cms.untracked.string("zGolden")
goodZToMuMuPathLoose.__iadd__(goodZToMuMuEdmNtupleLoose)
goodZToMuMuPathLoose.setLabel("goodZToMuMuEdmLoose")


#goodZToMuMuSameChargeEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
#goodZToMuMuSameChargeEdmNtupleLoose.src = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLTLoose")
#goodZToMuMuSameChargeEdmNtupleLoose.prefix = cms.untracked.string("zSameCharge")
#goodZToMuMuSameChargeEdmNtupleLoose.variables += zGolden
#goodZToMuMuSameChargePathLoose.__iadd__(goodZToMuMuSameChargeEdmNtupleLoose)
#goodZToMuMuSameChargePathLoose.setLabel("goodZToMuMuSameChargeLoose")



#goodZToMuMuOneStandAloneEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
#goodZToMuMuOneStandAloneEdmNtupleLoose.src=cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLTLoose")
#goodZToMuMuOneStandAloneEdmNtupleLoose.prefix=cms.untracked.string("zMuSta")
#goodZToMuMuOneStandAloneEdmNtupleLoose.variables += zGolden
#goodZToMuMuOneStandAloneMuonPathLoose.__iadd__(goodZToMuMuOneStandAloneEdmNtupleLoose)
#goodZToMuMuOneStandAloneMuonPathLoose.setLabel("goodZToMuMuOneStandAloneMuonLoose")

#goodZToMuMuOneTrackEdmNtupleLoose= copy.deepcopy(goodZToMuMuEdmNtuple)
#goodZToMuMuOneTrackEdmNtupleLoose.src=cms.InputTag("goodZToMuMuOneTrackFirstHLTLoose")
#goodZToMuMuOneTrackEdmNtupleLoose.prefix=cms.untracked.string("zMuTrk")
#goodZToMuMuOneTrackEdmNtupleLoose.variables += zMuTrk
#goodZToMuMuOneTrackPathLoose.__iadd__(goodZToMuMuOneTrackEdmNtupleLoose)
#goodZToMuMuOneTrackPathLoose.setLabel("goodZToMuMuOneTrackLoose")

ntuplesOut = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('NtupleLooseTestNew.root'),
    outputCommands = cms.untracked.vstring(
      "drop *",
      "keep *_goodZToMuMuEdmNtupleLoose_*_*",
      #"keep *_goodZToMuMuSameChargeEdmNtupleLoose_*_*",
      #"keep *_goodZToMuMuOneStandAloneEdmNtupleLoose_*_*",
      #"keep *_goodZToMuMuOneTrackEdmNtupleLoose_*_*"
      
    )
    )


ntuplesOut.setLabel("ntuplesOut")
NtuplesOut.__iadd__(ntuplesOut)
NtuplesOut.setLabel("NtuplesOut")


